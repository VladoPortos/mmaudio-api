# MMAudio API

A FastAPI-based REST API for the MMAudio model that generates high-quality synchronized audio for videos. This API provides a simple interface to process videos with MMAudio and returns the processed video in the response.

---

## ☕ Buy Me a Coffee (or a Beer!)

If you like this project and want to support my caffeine-fueled coding sessions, you can buy me a coffee (or a beer, I won't judge! 🍻) on Ko-fi:

[![Support me on Ko-fi](img/support_me_on_kofi_badge_red.png)](https://ko-fi.com/vladoportos)

Every donation helps to prove to my wife that I'm not a complete idiot :D

---

## Features

- Upload videos and generate synchronized audio using MMAudio
- Synchronous processing that returns the processed video directly in the response
- Docker and Docker Compose integration for easy deployment
- Supports all MMAudio model variants
- Customizable audio generation parameters (prompts, strength, steps, etc.)
- Automatic cleanup of temporary files

**Note:**
- The largest model can be run on a GPU with 16GB VRAM GPU (RTX 4070ti), and take about 60sec to generate 8sec of audio, but it also eats a lot of RAM like 25GB+
- Docker image will build it self to around 18.5GB too, so its not for the faint of heart

## Requirements

- Docker and Docker Compose
- NVIDIA GPU with CUDA support (for optimal performance)
- NVIDIA Container Toolkit (for GPU support in containers)

## Quick Start

1. Clone this repository:
   ```bash
   git clone https://github.com/YourUsername/mmaudio-api.git
   cd mmaudio-api
   ```

2. Build and start the container:
   ```bash
   docker-compose up -d
   ```

3. The API will be available at http://localhost:8000

## API Endpoints

### POST /process

Upload a video and process it with MMAudio to generate synchronized audio. The processed video is returned directly in the response.

**Form Parameters:**
- `video`: The video file to process (multipart/form-data)
- `prompt`: Text prompt for audio generation (optional)
- `negative_prompt`: Negative text prompt (optional)
- `duration`: Duration in seconds (default: 8.0)
- `cfg_strength`: Guidance scale (default: 4.5)
- `num_steps`: Number of generation steps (default: 25)
- `variant`: Model variant (default: "large_44k_v2")

**Response:**
- The processed video file with synchronized audio (Content-Type: video/mp4)

**Note:** Processing may take some time depending on the video length and model complexity.

## Model Variants

MMAudio comes with several model variants that offer different trade-offs between quality, processing speed, and memory requirements:

| Variant | Size | Audio Quality | Sampling Rate | Memory Usage | Relative Speed | Description |
|---------|------|---------------|--------------|--------------|----------------|-------------|
| `small_16k` | Small | Good | 16 kHz | ~2-3 GB | Fastest | Good for quick tests, lower quality audio |
| `small_44k` | Small | Better | 44 kHz | ~3-4 GB | Fast | Better quality than 16kHz, still relatively fast |
| `medium_44k` | Medium | Very Good | 44 kHz | ~4-6 GB | Medium | Good balance between quality and speed |
| `large_44k` | Large | Excellent | 44 kHz | ~6-8 GB | Slow | High-quality audio output |
| `large_44k_v2` | Large | Best | 44 kHz | ~6-8 GB | Slowest | Improved version of large_44k, highest quality (Default) |

### Key Differences

- **Sampling Rate**: 
  - 16kHz models produce audio with lower fidelity but process faster
  - 44kHz models produce higher-quality audio with better frequency response

- **Model Size**:
  - Small models: Faster processing, lower memory requirements, but less accurate audio synthesis
  - Medium models: Balance between performance and quality
  - Large models: Highest quality audio generation, but require more processing time and memory

- **When to use each variant**:
  - Use `small_16k` for quick tests or when resources are limited
  - Use `small_44k` or `medium_44k` for faster processing with reasonable quality
  - Use `large_44k` or `large_44k_v2` (default) for production-quality audio

To specify a model variant, use the `variant` parameter in your API request.

## Usage with Postman

Here is a picture of how to use the API with Postman:

![Postman](ing/postman.png)



## Development

To run the application locally without Docker:

1. Install the MMAudio repository:
   ```bash
   git clone https://github.com/hkchengrex/MMAudio.git
   cd MMAudio
   pip install -e .
   cd ..
   ```

2. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the API:
   ```bash
   python main.py
   ```

## License

This project is based on MMAudio, which is subject to its own license terms. Please refer to the [MMAudio repository](https://github.com/hkchengrex/MMAudio) for licensing information.
