# Text to Image Generator

A web application that generates images from text descriptions using the Flux AI API.

## Local Setup

1. Install requirements:
```bash
pip install -r requirements.txt
```

2. Set up environment variables:
- Create a `.env` file with your Flux API key:
```
FLUX_API_KEY=your_flux_api_key_here
```

3. Run the application:
```bash
python app.py
```

## Google Colab Setup

1. Upload the `colab_app.ipynb` to Google Colab
2. Replace `your_flux_api_key_here` with your actual API key
3. Run all cells in order
4. Click on the ngrok URL provided in the output

## Features

- Text to image generation
- Real-time progress indicator
- Error handling
- Mobile-responsive UI
- Support for negative prompts
- Customizable generation parameters

## Parameters

- samples: Number of images to generate
- steps: Number of denoising steps (default: 30)
- cfg_scale: Classifier Free Guidance scale (default: 7.5)
- negative_prompt: What to avoid in the generated image
