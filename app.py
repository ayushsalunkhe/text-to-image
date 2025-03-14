from flask import Flask, render_template, request, jsonify, send_file
from together import Together
from pyngrok import ngrok
from huggingface_hub import login, InferenceClient
import logging
import os
import base64
from io import BytesIO
import time
import random
from functools import wraps
from translate import Translator
from datetime import datetime
import re
import cv2
import numpy as np
from PIL import Image
import torch
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
import langdetect
from langdetect import DetectorFactory, LangDetectException
import requests

# Initialize Flask app
app = Flask(__name__)

# Set up logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create output directory if it doesn't exist
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Set fixed seed for language detection to ensure consistent results
DetectorFactory.seed = 0

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

translator = Translator(to_lang="en")

# Configure Gemini API
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)

@retry_with_backoff(retries=3, backoff_in_seconds=1)
def detect_language_with_gemini(text):
    """Detect language using Gemini API"""
    try:
        model = genai.GenerativeModel('gemini-2.0-flash-thinking-exp-01-21')
        prompt = f"""
        Detect the language of the following text. 
        Return only the ISO 639-1 language code (e.g., 'en' for English, 'es' for Spanish, etc.).
        
        Text: {text}
        """
        
        response = model.generate_content(prompt)
        language_code = response.text.strip().lower()
        
        # Clean up response to ensure it's just the language code
        language_code = re.sub(r'[^a-z]', '', language_code)
        
        # Fallback to langdetect if Gemini returns invalid code
        if len(language_code) != 2:
            logger.warning(f"Gemini returned invalid language code: {language_code}. Falling back to langdetect.")
            return langdetect.detect(text)
            
        return language_code
    except Exception as e:
        logger.error(f"Gemini language detection error: {str(e)}")
        # Fallback to langdetect
        return langdetect.detect(text)

@retry_with_backoff(retries=3, backoff_in_seconds=1)
def translate_with_gemini(text, target_lang='en'):
    """Translate text using Gemini API"""
    try:
        # First detect the language
        source_lang = detect_language_with_gemini(text)
        
        # If already in target language, return as is
        if source_lang == target_lang:
            return text, source_lang
            
        model = genai.GenerativeModel('gemini-2.0-flash-thinking-exp-01-21')
        prompt = f"""
        Translate the following text from {source_lang} to {target_lang}.
        Return only the translated text without any additional explanations.
        
        Text: {text}
        """
        
        response = model.generate_content(prompt)
        translated_text = response.text.strip()
        
        return translated_text, source_lang
    except Exception as e:
        logger.error(f"Gemini translation error: {str(e)}")
        # Fallback to basic translator
        try:
            translation = translator.translate(text)
            return translation, 'unknown'
        except:
            return text, 'unknown'

@retry_with_backoff(retries=3, backoff_in_seconds=1)
def translate_text(text, target_lang='en'):
    """Translate text to target language with Gemini fallback"""
    try:
        # Try with Gemini first
        return translate_with_gemini(text, target_lang)
    except Exception as e:
        logger.error(f"Gemini translation failed: {str(e)}")
        # Fallback to basic translator
        try:
            # Detect the source language
            source_lang = langdetect.detect(text)
            
            # If text is already in English, return as is
            if source_lang == target_lang:
                return text, source_lang
                
            # Translate using translate package
            translation = translator.translate(text)
            return translation, source_lang
        except Exception as e:
            logger.error(f"Translation error: {str(e)}")
            return text, 'unknown'  # Return original text if translation fails

@app.route('/detect-language', methods=['POST'])
def detect_language():
    """Detect the language of input text using Gemini"""
    try:
        text = request.form.get('text')
        if not text:
            return jsonify({'success': False, 'error': 'No text provided'}), 400
            
        # Try Gemini first, fallback to langdetect
        try:
            language = detect_language_with_gemini(text)
        except Exception:
            language = langdetect.detect(text)
            
        # Get language name
        try:
            language_name = langdetect.langdetect.detect_lang_name(language)
        except:
            # Fallback language names for common codes
            language_names = {
                'en': 'English', 'es': 'Spanish', 'fr': 'French', 'de': 'German',
                'it': 'Italian', 'pt': 'Portuguese', 'ru': 'Russian', 'ja': 'Japanese',
                'zh': 'Chinese', 'ko': 'Korean', 'ar': 'Arabic', 'hi': 'Hindi'
            }
            language_name = language_names.get(language, 'Unknown')
        
        return jsonify({
            'success': True, 
            'language': language,
            'language_name': language_name
        })
    except Exception as e:
        logger.error(f"Language detection error: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/translate', methods=['POST'])
def translate():
    """Translate text using Gemini"""
    try:
        text = request.form.get('text')
        target_lang = request.form.get('target_lang', 'en')
        
        if not text:
            return jsonify({'success': False, 'error': 'No text provided'}), 400
            
        translated_text, source_lang = translate_with_gemini(text, target_lang)
        return jsonify({
            'success': True,
            'translated_text': translated_text,
            'source_language': source_lang
        })
    except Exception as e:
        logger.error(f"Translation error: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

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

# Initialize clients
client = Together()
hf_client = InferenceClient(provider="hf-inference", api_key=HF_TOKEN)

# Update API URLs and add new model
API_URL = "https://api-inference.huggingface.co/models/black-forest-labs/FLUX.1-dev"
SD_MODEL = "stabilityai/stable-diffusion-3.5-large-turbo"
headers = {"Authorization": f"Bearer {os.environ['HUGGINGFACE_TOKEN']}"}

@retry_with_backoff(retries=3, backoff_in_seconds=1)
def sync_enhance_prompt_with_gemini(prompt, preserve_language=True):
    """Enhance the user's prompt using Gemini 2.0 Flash with multilingual support"""
    try:
        # First detect the original language
        original_lang = detect_language_with_gemini(prompt)
        
        # Translate to English for processing if not already in English
        if original_lang != 'en':
            translated_prompt, _ = translate_with_gemini(prompt, 'en')
        else:
            translated_prompt = prompt
        
        # Configure the model
        model = genai.GenerativeModel('gemini-2.0-flash-thinking-exp-01-21')
        
        # Create a structured prompt for Gemini using more neutral language
        gemini_prompt = f"""
        Please help improve this image description by:
        1. Adding visual details (lighting, composition, perspective)
        2. Including style elements (artistic style, mood, atmosphere)
        3. Enhancing descriptive language (colors, textures, materials)
        4. Maintaining the original meaning and intent
        
        Input description: {translated_prompt}
        
        Provide only the enhanced description without any additional text or explanations.
        """
        
        # Generate the enhanced prompt with adjusted safety settings
        response = model.generate_content(
            gemini_prompt,
            safety_settings=[
                {
                    "category": "HARM_CATEGORY_HARASSMENT",
                    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                },
                {
                    "category": "HARM_CATEGORY_HATE_SPEECH",
                    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                },
                {
                    "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                },
                {
                    "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                }
            ]
        )
        
        # Extract the enhanced prompt
        enhanced_prompt = response.text.strip()
        
        # If the original prompt was not in English and we want to preserve language,
        # translate the enhanced prompt back to the original language
        if original_lang != 'en' and preserve_language:
            enhanced_prompt, _ = translate_with_gemini(enhanced_prompt, original_lang)
            
        logger.debug(f"Enhanced prompt: {enhanced_prompt}")
        return enhanced_prompt
    except Exception as e:
        logger.error(f"Error enhancing prompt with Gemini: {str(e)}")
        return prompt  # Fallback to original prompt if enhancement fails

@retry_with_backoff(retries=3, backoff_in_seconds=1)
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

@retry_with_backoff(retries=3)
def generate_with_sd(prompt):
    """Generate image using Stable Diffusion 3.5"""
    logger.info("Generating image with Stable Diffusion 3.5")
    try:
        # Generate image
        image = hf_client.text_to_image(
            prompt,
            model=SD_MODEL
        )
        
        # Convert PIL Image to bytes
        img_byte_arr = BytesIO()
        image.save(img_byte_arr, format='PNG')
        image_bytes = img_byte_arr.getvalue()
        
        # Save image locally
        save_image_locally(image_bytes, prompt, "stable-diffusion")
        
        # Convert to base64
        image_base64 = base64.b64encode(image_bytes).decode('utf-8')
        logger.info("Successfully generated image with Stable Diffusion 3.5")
        return image_base64
        
    except Exception as e:
        logger.error(f"Stable Diffusion generation error: {str(e)}")
        raise

@retry_with_backoff(retries=3)
def generate_with_gemini(prompt):
    """Generate image using Gemini API"""
    logger.info("Generating image with Gemini")
    try:
        # Configure the model - use gemini-1.5-pro for image generation
        model = genai.GenerativeModel('gemini-2.0-flash-thinking-exp-01-21')
        
        # Generate image
        response = model.generate_content(
            f"Generate an image based on this description: {prompt}",
            generation_config={"temperature": 0.9}
        )
        
        # Extract image data
        if response.parts and hasattr(response.parts[0], 'image_data'):
            image_bytes = response.parts[0].image_data
            
            # Save image locally
            save_image_locally(image_bytes, prompt, "gemini")
            
            # Convert to base64
            image_base64 = base64.b64encode(image_bytes).decode('utf-8')
            logger.info("Successfully generated image with Gemini")
            return image_base64
        else:
            raise Exception("No image data in Gemini response")
            
    except Exception as e:
        logger.error(f"Gemini image generation error: {str(e)}")
        raise

def init_realesrgan():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
    
    # Create weights directory if it doesn't exist
    model_path = os.path.join(os.path.dirname(__file__), 'weights')
    os.makedirs(model_path, exist_ok=True)
    
    # Model path
    model_file = os.path.join(model_path, 'RealESRGAN_x4plus.pth')
    
    # Download model if not exists
    if not os.path.exists(model_file):
        url = 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth'
        response = requests.get(url)
        with open(model_file, 'wb') as f:
            f.write(response.content)
    
    # Initialize upsampler
    upsampler = RealESRGANer(
        scale=4,
        model_path=model_file,
        model=model,
        tile=512,  # Increased tile size for better quality
        tile_pad=32,  # Increased padding to reduce artifacts
        pre_pad=0,
        half=device == 'cuda'  # Use half precision on CUDA
    )
    
    return upsampler

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


@app.route('/upscale', methods=['GET', 'POST'])
def upscale():
    try:
        if request.method == 'POST':
            if 'image' not in request.files:
                return 'No image uploaded', 400
            
            file = request.files['image']
            if file.filename == '':
                return 'No image selected', 400

            # Read the image
            img_stream = BytesIO(file.read())
            original_image = Image.open(img_stream).convert('RGB')
            
            # Convert PIL Image to numpy array
            input_img = np.array(original_image)
            
            try:
                # Initialize Real-ESRGAN
                upsampler = init_realesrgan()
                
                # Process the image with tiling
                output, _ = upsampler.enhance(input_img, outscale=4)
                
                # Convert output to PIL Image
                upscaled_pil = Image.fromarray(output)
                
            except Exception as e:
                logging.error(f"Real-ESRGAN failed, falling back to OpenCV: {str(e)}")
                # Fallback to OpenCV if Real-ESRGAN fails
                # Use LANCZOS4 for high-quality upscaling
                upscaled = cv2.resize(input_img, None, fx=4, fy=4, interpolation=cv2.INTER_LANCZOS4)
                upscaled_pil = Image.fromarray(upscaled)
            
            # Convert images to base64 for display
            buffered = BytesIO()
            original_image.save(buffered, format="PNG", optimize=True)
            original_base64 = base64.b64encode(buffered.getvalue()).decode()
            
            buffered = BytesIO()
            upscaled_pil.save(buffered, format="PNG", optimize=True)
            upscaled_base64 = base64.b64encode(buffered.getvalue()).decode()
            
            return render_template('upscale.html', 
                                original_image=original_base64,
                                upscaled_image=upscaled_base64)
        
        return render_template('upscale.html')
    except Exception as e:
        logging.error(f"Error in upscale route: {str(e)}")
        return "An error occurred while processing the image", 500

@app.route('/enhance-prompt', methods=['POST'])
def enhance_prompt():
    try:
        prompt = request.form.get('prompt')
        preserve_language = request.form.get('preserve_language', 'true').lower() == 'true'
        
        if not prompt:
            return jsonify({'success': False, 'error': 'No prompt provided'}), 400
            
        enhanced_prompt = sync_enhance_prompt_with_gemini(prompt, preserve_language)
        
        # Detect languages for response
        original_lang = detect_language_with_gemini(prompt)
        enhanced_lang = detect_language_with_gemini(enhanced_prompt)
        
        return jsonify({
            'success': True, 
            'enhanced_prompt': enhanced_prompt,
            'original_language': original_lang,
            'enhanced_language': enhanced_lang
        })
    except Exception as e:
        logger.error(f"Error in enhance_prompt endpoint: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/generate', methods=['POST'])
def generate_image():
    try:
        prompt = request.form.get('prompt')
        model = request.form.get('model')
        preserve_language = request.form.get('preserve_language', 'false').lower() == 'true'
        
        if not prompt or not model:
            logger.error("Missing required parameters")
            return jsonify({'success': False, 'error': 'Missing prompt or model'}), 400
            
        logger.debug(f"Received prompt: {prompt} and model: {model}")
        
        # Detect original language
        original_lang = detect_language_with_gemini(prompt)
        
        # Enhance the prompt using Gemini
        enhanced_prompt = sync_enhance_prompt_with_gemini(prompt, preserve_language)
        
        # For non-English prompts, we need to translate to English for most models
        # unless preserve_language is True and the model supports it
        if original_lang != 'en' and not preserve_language:
            # Translate to English for image generation
            generation_prompt, _ = translate_with_gemini(enhanced_prompt, 'en')
        else:
            generation_prompt = enhanced_prompt
        
        try:
            if model == "gemini":
                # Use Gemini for image generation
                image_data = generate_with_gemini(generation_prompt)
            elif model == "black-forest-labs/FLUX.1-dev":
                # Use Hugging Face API
                logger.debug("Making API request to Hugging Face")
                image_data = generate_with_huggingface(generation_prompt)
            elif model == SD_MODEL:
                # Use Stable Diffusion 3.5
                logger.debug("Making API request to Stable Diffusion")
                image_data = generate_with_sd(generation_prompt)
            else:
                # Use Together AI
                logger.debug("Making API request to Together AI")
                image_data = generate_with_together(generation_prompt, model)
                
            return jsonify({
                'success': True, 
                'image': image_data,
                'original_language': original_lang,
                'used_prompt': generation_prompt,
                'enhanced_prompt': enhanced_prompt
            })
                
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
