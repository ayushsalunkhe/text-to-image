from flask import Flask, render_template, request, jsonify
from together import Together
from pyngrok import ngrok
import logging
import os

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Set Together API key
os.environ["TOGETHER_API_KEY"] = "1dd1a7e6cbd43070e903346ae2638952d71e074221fed5deabb4869232499fbd"

app = Flask(__name__)
# Initialize Together client
client = Together()

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
    prompt = request.form.get('prompt')
    logger.debug(f"Received prompt: {prompt}")
    
    try:
        logger.debug("Making API request to Together AI")
        response = client.images.generate(
            prompt=prompt,
            model="black-forest-labs/FLUX.1-schnell",
            width=1024,
            height=768,
            steps=4,
            n=1,
            response_format="b64_json"
        )
        
        if response and response.data:
            image_data = response.data[0].b64_json
            return jsonify({'success': True, 'image': image_data})
        else:
            logger.error("No image data in response")
            return jsonify({'success': False, 'error': 'No image generated'})
            
    except Exception as e:
        logger.error(f"Exception occurred: {str(e)}", exc_info=True)
        return jsonify({'success': False, 'error': str(e)})

if __name__ == '__main__':
    # Set your ngrok auth token
    ngrok.set_auth_token("2kF0h1FF3cshznbJqIAa7uaBpUd_5ouG3nr6T6Br8NYF9kuV")
    public_url = ngrok.connect(5000).public_url
    print(f" * ngrok tunnel \"{public_url}\" -> \"http://127.0.0.1:5000\"")
    app.run(debug=False)
