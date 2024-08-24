import logging
from utils import query_llm_api
from language_utils import LanguageDetector

# Configure logging
logging.basicConfig(level=logging.INFO)

class PerspectiveAgent:
    def __init__(self, model_path='C:/Users/manus/fastText/lid.176.bin'):
        self.language_detector = LanguageDetector(model_path)
        self.DEFAULT_LANGUAGE = 'en_XX'

    def generate_perspectives(self, topic, conversation_history):
        sanitized_topic = self.sanitize_input(topic)
        formatted_history = self.format_history(conversation_history)

        # Use both topic and conversation history for language detection
        detected_lang = self.detect_language_with_history(sanitized_topic, formatted_history)

        # Handle ambiguity in the topic
        if self.is_ambiguous(sanitized_topic):
            follow_up_question = self.get_follow_up_question(sanitized_topic)
            logging.info(f"Generated follow-up question to clarify ambiguity: {follow_up_question}")
            return follow_up_question
        
        # Generate perspectives based on the topic and conversation context
        prompt = self.create_prompt(sanitized_topic, formatted_history)

        try:
            logging.info(f"Sending prompt for perspectives on '{sanitized_topic}' in language: {detected_lang}")
            response = query_llm_api(prompt, detected_lang)
            logging.info(f"Received perspectives: {response}")
            return response
        except Exception as e:
            logging.error(f"Error generating perspectives for topic '{sanitized_topic}': {str(e)}")
            return f"An error occurred while generating perspectives: {str(e)}"

    def detect_language_with_history(self, topic, conversation_history):
        # Try detecting language based on the conversation history first, then fallback to topic
        detected_lang = self.language_detector.detect_language_ft(conversation_history) or \
                        self.language_detector.detect_language_ft(topic) or \
                        self.DEFAULT_LANGUAGE
        logging.info(f"Detected language: {detected_lang}")
        return detected_lang

    def format_history(self, conversation_history):
        formatted_history = "\n".join([f"{entry['role'].capitalize()}: {entry['content']}" for entry in conversation_history])
        return formatted_history.strip()

    def is_ambiguous(self, topic):
        ambiguous_keywords = ['this', 'that', 'it', 'those', 'these']
        return any(keyword in topic.lower() for keyword in ambiguous_keywords)

    def get_follow_up_question(self, topic):
        return f"Could you please clarify what you mean by '{topic}'?"

    def create_prompt(self, topic, conversation_history):
        # Structure the prompt more clearly for generating perspectives
        return (
            f"Your task is to assemble a diverse panel of experts who will work together to offer a well-rounded response on the topic '{topic}'. "
            f"Each expert should bring a distinct perspective or expertise relevant to this topic. "
            f"Here is the ongoing conversation for context:\n{conversation_history}\n"
            f"For each expert, please include a brief description of their area of expertise, and suggest any insights that would help in thoroughly exploring this topic."
        )

    @staticmethod
    def sanitize_input(text):
        return text.strip()

# Example usage
perspective_agent = PerspectiveAgent()
