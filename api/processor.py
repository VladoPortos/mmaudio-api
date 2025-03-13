import os
import logging
import torch
import torchaudio
import gc
from pathlib import Path

from mmaudio.eval_utils import (ModelConfig, all_model_cfg, generate, load_video, make_video,
                              setup_eval_logging)
from mmaudio.model.flow_matching import FlowMatching
from mmaudio.model.networks import MMAudio, get_my_mmaudio
from mmaudio.model.utils.features_utils import FeaturesUtils

# Enable TF32 precision for better performance on NVIDIA GPUs
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
# Enable benchmark mode for faster performance with fixed input sizes
torch.backends.cudnn.benchmark = True

logger = logging.getLogger(__name__)
setup_eval_logging()

# Helper function for memory cleanup
def cleanup_gpu_memory():
    """Clean up GPU memory and trigger garbage collection"""
    if torch.cuda.is_available():
        # Empty CUDA cache
        torch.cuda.empty_cache()
        # Force garbage collection
        gc.collect()
        logger.info(f"Memory after cleanup: {torch.cuda.max_memory_allocated() / (2**30):.2f} GB")

@torch.inference_mode()
def process_video(
    task_id: str,
    video_path: str,
    prompt: str = "",
    negative_prompt: str = "",
    duration: float = 8.0,
    cfg_strength: float = 4.5,
    num_steps: int = 25,
    variant: str = "large_44k_v2",
    output_dir: str = "./output",
    mask_away_clip: bool = False,
    full_precision: bool = False,
    seed: int = 42,
):
    """
    Process the video using MMAudio model.
    
    Args:
        task_id: Unique identifier for the task
        video_path: Path to the input video file
        prompt: Input prompt for audio generation
        negative_prompt: Negative prompt for audio generation
        duration: Duration for audio generation in seconds
        cfg_strength: Guidance scale for classifier-free guidance
        num_steps: Number of steps for generation
        variant: Model variant (e.g., 'large_44k_v2')
        output_dir: Output directory for generated files
        mask_away_clip: If True, mask away CLIP features
        full_precision: If True, use full precision (float32)
        seed: Random seed for generation
        
    Returns:
        dict: Paths to generated files
    """
    try:
        # Clean up GPU memory from previous runs
        cleanup_gpu_memory()
        
        logger.info(f"Processing video for task {task_id}")
        
        # Check if variant exists
        if variant not in all_model_cfg:
            raise ValueError(f"Unknown model variant: {variant}")
        
        # Get model config and download if needed
        model: ModelConfig = all_model_cfg[variant]
        model.download_if_needed()
        seq_cfg = model.seq_cfg
        
        # Convert paths to Path objects
        video_path = Path(video_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set device and dtype
        device = 'cpu'
        if torch.cuda.is_available():
            device = 'cuda'
        elif torch.backends.mps.is_available():
            device = 'mps'
        else:
            logger.warning('CUDA/MPS are not available, running on CPU')
        
        # Performance optimization: Use float16 for large models on CUDA (only for models, not solver)
        is_large_model = 'large' in variant
        dtype = torch.float32
        if not full_precision and device == 'cuda':
            dtype = torch.float16  # Use float16 instead of bfloat16 for better CUDA performance
        elif not full_precision:
            dtype = torch.bfloat16
        
        # Performance optimization: Reduce steps for large model 
        # Stay with 'euler' mode which is more stable
        inference_mode = 'euler'
        original_steps = num_steps
        if is_large_model and device == 'cuda':
            # For large_44k_v2, we can reduce steps which significantly improves performance
            if num_steps == 25 and variant == 'large_44k_v2':
                logger.info("Optimizing: Reducing steps from 25 to 15 for large_44k_v2 model")
                num_steps = 15
        
        logger.info(f"Using inference mode: {inference_mode} with {num_steps} steps (originally requested: {original_steps})")
        
        # Load model
        net: MMAudio = get_my_mmaudio(model.model_name).to(device, dtype).eval()
        net.load_weights(torch.load(model.model_path, map_location=device, weights_only=True))
        logger.info(f"Loaded weights from {model.model_path}")
        
        # Performance optimization: Use torch.compile if PyTorch 2.0+
        # Only try to compile the model, not anything related to ODE solving
        if hasattr(torch, 'compile') and device == 'cuda':
            try:
                logger.info("Optimizing: Using torch.compile() for model")
                net = torch.compile(net)
            except Exception as e:
                logger.warning(f"Failed to compile model: {str(e)}. Continuing with uncompiled model.")
        
        # Setup for generation
        rng = torch.Generator(device=device)
        rng.manual_seed(seed)
        fm = FlowMatching(min_sigma=0, inference_mode=inference_mode, num_steps=num_steps)
        
        feature_utils = FeaturesUtils(
            tod_vae_ckpt=model.vae_path,
            synchformer_ckpt=model.synchformer_ckpt,
            enable_conditions=True,
            mode=model.mode,
            bigvgan_vocoder_ckpt=model.bigvgan_16k_path,
            need_vae_encoder=False
        )
        feature_utils = feature_utils.to(device, dtype).eval()
        
        # Load video
        logger.info(f"Using video {video_path}")
        video_info = load_video(video_path, duration)
        clip_frames = video_info.clip_frames
        sync_frames = video_info.sync_frames
        duration = video_info.duration_sec
        
        if mask_away_clip:
            clip_frames = None
        else:
            clip_frames = clip_frames.unsqueeze(0)
        
        sync_frames = sync_frames.unsqueeze(0)
        
        # Set sequence configuration
        seq_cfg.duration = duration
        net.update_seq_lengths(seq_cfg.latent_seq_len, seq_cfg.clip_seq_len, seq_cfg.sync_seq_len)
        
        logger.info(f"Prompt: {prompt}")
        logger.info(f"Negative prompt: {negative_prompt}")
        
        # Generate audio using the standard generate function with mixed precision if on CUDA
        if device == 'cuda' and not full_precision:
            logger.info("Optimizing: Using torch.amp.autocast for mixed precision")
            with torch.amp.autocast('cuda'):
                audios = generate(
                    clip_frames,
                    sync_frames, 
                    [prompt],
                    negative_text=[negative_prompt],
                    feature_utils=feature_utils,
                    net=net,
                    fm=fm,
                    rng=rng,
                    cfg_strength=cfg_strength
                )
        else:
            # We can use it directly now with inference_mode decorator
            audios = generate(
                clip_frames,
                sync_frames, 
                [prompt],
                negative_text=[negative_prompt],
                feature_utils=feature_utils,
                net=net,
                fm=fm,
                rng=rng,
                cfg_strength=cfg_strength
            )
        
        audio = audios.float().cpu()[0]
        
        # Save output audio
        audio_save_path = output_dir / f"{video_path.stem}.flac"
        torchaudio.save(audio_save_path, audio, seq_cfg.sampling_rate)
        logger.info(f"Audio saved to {audio_save_path}")
        
        # Create and save output video with audio
        video_save_path = output_dir / f"{video_path.stem}.mp4"
        make_video(video_info, video_save_path, audio, sampling_rate=seq_cfg.sampling_rate)
        logger.info(f"Video saved to {video_save_path}")
        
        # Log memory usage if CUDA available
        if torch.cuda.is_available():
            logger.info(f"Memory usage: {torch.cuda.max_memory_allocated() / (2**30):.2f} GB")
        
        # Explicitly clean up tensors that might hold references
        if device == 'cuda':
            del audios, audio, net, feature_utils, fm, clip_frames, sync_frames
            # Clean up GPU memory again
            cleanup_gpu_memory()
        
        # Return paths to generated files
        return {
            "audio_path": str(audio_save_path),
            "video_path": str(video_save_path)
        }
    
    except Exception as e:
        # Make sure to clean up even if there's an error
        if torch.cuda.is_available():
            cleanup_gpu_memory()
        logger.exception(f"Error processing video for task {task_id}: {str(e)}")
        raise
