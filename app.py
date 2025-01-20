from flask import Flask, render_template, request, jsonify
from together import Together
from pyngrok import ngrok
from huggingface_hub import login
import logging
import os
import base64
from io import BytesIO
import time
import random
from functools import wraps
import requests
from datetime import datetime
import re

# Set up logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create output directory if it doesn't exist
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)

def sanitize_filename(prompt):
    """Convert prompt to a valid filename by removing special characters and limiting length"""
    # Remove special characters and replace spaces with underscores
    filename = re.sub(r'[^\w\s-]', '', prompt)
    filename = re.sub(r'[-\s]+', '_', filename).strip('-_')
    # Limit length to 50 characters
    return filename[:50]

def save_image_locally(image_bytes, prompt, model_name):
    """Save the generated image to the output directory"""
    try:
        # Create a timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        # Sanitize the prompt for use in filename
        safe_prompt = sanitize_filename(prompt)
        # Create filename with timestamp and prompt
        filename = f"{timestamp}_{safe_prompt}.png"
        filepath = os.path.join(OUTPUT_DIR, filename)
        
        # Save the image
        with open(filepath, 'wb') as f:
            f.write(image_bytes)
        
        logger.info(f"Image saved successfully at: {filepath}")
        return filepath
    except Exception as e:
        logger.error(f"Error saving image locally: {str(e)}", exc_info=True)
        return None

def retry_with_backoff(retries=3, backoff_in_seconds=1):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            x = 0
            while True:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if x == retries:
                        logger.error(f"All {retries} retry attempts failed")
                        raise
                    wait_time = (backoff_in_seconds * 2 ** x + 
                               random.uniform(0, 1))
                    logger.warning(f"Attempt {x + 1} failed: {str(e)}. Retrying in {wait_time:.2f} seconds...")
                    time.sleep(wait_time)
                    x += 1
        return wrapper
    return decorator

# Set API keys
os.environ["TOGETHER_API_KEY"] = "1dd1a7e6cbd43070e903346ae2638952d71e074221fed5deabb4869232499fbd"
HF_TOKEN = "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"
os.environ["HUGGINGFACE_TOKEN"] = HF_TOKEN

# Login to Hugging Face
try:
    login(token=HF_TOKEN)
    logger.info("Successfully logged in to Hugging Face")
except Exception as e:
    logger.error(f"Failed to login to Hugging Face: {str(e)}")

app = Flask(__name__)
# Initialize Together client
client = Together()

API_URL = "https://api-inference.huggingface.co/models/black-forest-labs/FLUX.1-dev"
headers = {"Authorization": f"Bearer {os.environ['HUGGINGFACE_TOKEN']}"}

@retry_with_backoff(retries=3)
def generate_with_huggingface(prompt):
    logger.info("Generating image with Hugging Face API")
    try:
        logger.debug(f"Sending request to Hugging Face API with prompt: {prompt}")
        response = requests.post(API_URL, headers=headers, json={"inputs": prompt})
        response.raise_for_status()
        logger.debug(f"Response status code: {response.status_code}")
        logger.debug(f"Response headers: {response.headers}")
        
        image_bytes = response.content
        logger.debug(f"Received image data of size: {len(image_bytes)} bytes")
        
        # Save image locally
        save_image_locally(image_bytes, prompt, "huggingface")
        
        # Convert bytes to base64
        image_base64 = base64.b64encode(image_bytes).decode('utf-8')
        logger.debug(f"Converted to base64 string of length: {len(image_base64)}")
        
        logger.info("Successfully generated image with Hugging Face API")
        return image_base64
    except requests.exceptions.RequestException as e:
        logger.error(f"Error calling Hugging Face API: {str(e)}", exc_info=True)
        raise Exception("Failed to generate image using Hugging Face API") from e

@retry_with_backoff(retries=2)  # Reduced retries for faster feedback
def generate_with_together(prompt, model):
    logger.info(f"Generating image with Together AI using model: {model}")
    try:
        response = client.images.generate(
            prompt=prompt,
            model=model,
            width=768,  # Reduced for faster generation
            height=768,
            steps=2,    # Reduced steps for faster generation
            n=1,
            response_format="b64_json"
        )
        if not response or not response.data:
            raise Exception("No image data in response")
            
        # Convert base64 to bytes and save locally
        image_bytes = base64.b64decode(response.data[0].b64_json)
        save_image_locally(image_bytes, prompt, "together")
        
        logger.info("Successfully generated image with Together AI")
        return response.data[0].b64_json
    except Exception as e:
        logger.error(f"Together AI generation error: {str(e)}")
        raise

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/about_us')
def about_us():
    return render_template('about_us.html')

@app.route('/contact_us')
def contact_us():
    return render_template('contact_us.html')

@app.route('/explore_tools')
def explore_tools():
    return render_template('explore_tools.html')

@app.route('/generate', methods=['POST'])
def generate_image():
    try:
        prompt = request.form.get('prompt')
        model = request.form.get('model')
        
        if not prompt or not model:
            logger.error("Missing required parameters")
            return jsonify({'success': False, 'error': 'Missing required parameters'}), 400
            
        logger.debug(f"Received prompt: {prompt} and model: {model}")
        
        try:
            if model == "black-forest-labs/FLUX.1-dev":
                # Use Hugging Face API
                logger.debug("Making API request to Hugging Face")
                image_data = generate_with_huggingface(prompt)
            else:
                # Use Together AI
                logger.debug("Making API request to Together AI")
                image_data = generate_with_together(prompt, model)
                
            return jsonify({'success': True, 'image': image_data})
                
        except Exception as e:
            error_msg = str(e)
            logger.error(f"API Error: {error_msg}", exc_info=True)
            return jsonify({'success': False, 'error': error_msg}), 500
            
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}", exc_info=True)
        return jsonify({'success': False, 'error': 'Internal server error'}), 500

if __name__ == '__main__':
    # Set your ngrok auth token
    ngrok.set_auth_token("2kF0h1FF3cshznbJqIAa7uaBpUd_5ouG3nr6T6Br8NYF9kuV")
    public_url = ngrok.connect(5000).public_url
    logger.info(f"Public URL: {public_url}")
    app.run()
