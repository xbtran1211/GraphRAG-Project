import logging
from langdetect import detect, LangDetectException
from huggingface_hub import InferenceClient

# Configure logging
logging.basicConfig(filename='chatbot_errors.log', level=logging.ERROR, format='%(asctime)s %(levelname)s %(message)s')

# Initialize the InferenceClient with API token and model name
client = InferenceClient(
    model="meta-llama/Meta-Llama-3-8B-Instruct",
    token="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"
)

def query_llm_api(text, detected_lang='en', max_tokens=100):
    try:
        logging.info(f"Sending request to LLM API with text: '{text}' in detected language: '{detected_lang}'")
        sanitized_text = text.strip()

        response = client.chat_completion(
            messages=[{"role": "user", "content": sanitized_text}],
            max_tokens=max_tokens,
            stream=False
        )

        if response and 'choices' in response and response['choices']:
            generated_text = response['choices'][0]['message']['content'].strip()
            logging.info(f"Generated text: {generated_text}")
            return generated_text
        else:
            error_msg = "No valid response received from the API."
            logging.error(error_msg)
            return "Sorry, I couldn't generate a response at the moment. Please try again."
    except Exception as e:
        logging.error(f"Error querying LLaMA model: {str(e)}")
        return "An unexpected error occurred. Please try again later."

def detect_language(text):
    try:
        logging.info(f"Detecting language for text: '{text}'")
        language = detect(text)
        logging.info(f"Detected language: {language}")
        return language
    except LangDetectException as e:
        logging.error(f"Language detection failed: {str(e)}. Defaulting to 'en'.")
        return 'en'
    except Exception as e:
        logging.error(f"Unexpected error during language detection: {str(e)}. Defaulting to 'en'.")
        return 'en'

