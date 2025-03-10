import os
import logging
import torch
import torchaudio
from pathlib import Path

from mmaudio.eval_utils import (ModelConfig, all_model_cfg, generate, load_video, make_video,
                              setup_eval_logging)
from mmaudio.model.flow_matching import FlowMatching
from mmaudio.model.networks import MMAudio, get_my_mmaudio
from mmaudio.model.utils.features_utils import FeaturesUtils

# Enable TF32 precision for better performance on NVIDIA GPUs
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

logger = logging.getLogger(__name__)
setup_eval_logging()

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
        
        dtype = torch.float32 if full_precision else torch.bfloat16
        
        # Load model
        net: MMAudio = get_my_mmaudio(model.model_name).to(device, dtype).eval()
        net.load_weights(torch.load(model.model_path, map_location=device, weights_only=True))
        logger.info(f"Loaded weights from {model.model_path}")
        
        # Setup for generation
        rng = torch.Generator(device=device)
        rng.manual_seed(seed)
        fm = FlowMatching(min_sigma=0, inference_mode='euler', num_steps=num_steps)
        
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
        
        # Generate audio using the standard generate function
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
        
        # Return paths to generated files
        return {
            "audio_path": str(audio_save_path),
            "video_path": str(video_save_path)
        }
    
    except Exception as e:
        logger.exception(f"Error processing video for task {task_id}: {str(e)}")
        raise
