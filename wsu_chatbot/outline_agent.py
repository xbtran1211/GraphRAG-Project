import logging
from utils import query_llm_api
from language_utils import LanguageDetector  # Ensure this class is properly implemented

# Configure logging
logging.basicConfig(level=logging.INFO)

class OutlineAgent:
    def __init__(self, model_path='C:/Users/manus/fastText/lid.176.bin'):
        # Initialize the language detector with a path to the model
        self.language_detector = LanguageDetector(model_path)
        self.DEFAULT_LANGUAGE = 'en_XX'  # Default language if detection fails

    def generate_outline(self, topic, conversation_history):
        """
        Generates an outline for the given topic, taking into account the conversation history.
        """
        # Sanitize the input topic and conversation history
        sanitized_topic = self.sanitize_input(topic)
        formatted_history = self.format_history(conversation_history)

        # Create the prompt using the sanitized inputs
        prompt = self.create_prompt(sanitized_topic, formatted_history)
        
        # Detect language based on the conversation history and the topic
        detected_lang = self.detect_language_with_fallback(formatted_history, sanitized_topic)
        
        # Handle ambiguity in the topic
        if self.is_ambiguous(sanitized_topic):
            follow_up_question = self.get_follow_up_question(sanitized_topic)
            logging.info(f"Generated follow-up question to clarify ambiguity: {follow_up_question}")
            return follow_up_question
        
        try:
            logging.info(f"Generating outline for '{sanitized_topic}' in language: {detected_lang}")
            response = query_llm_api(prompt, detected_lang)
            logging.info(f"Received outline: {response}")
            return response
        except Exception as e:
            logging.error(f"Error in generating outline for '{sanitized_topic}': {str(e)}")
            return f"An error occurred while generating an outline: {str(e)}"

    def detect_language_with_fallback(self, conversation_history, topic):
        """
        Detects the language based on the conversation history or falls back to the topic if necessary.
        """
        detected_lang = self.language_detector.detect_language_ft(conversation_history) or \
                        self.language_detector.detect_language_ft(topic) or \
                        self.DEFAULT_LANGUAGE
        logging.info(f"Detected language: {detected_lang}")
        return detected_lang

    def is_ambiguous(self, topic):
        """
        Check if the topic is ambiguous and may require a clarifying follow-up question.
        """
        ambiguous_keywords = ['this', 'that', 'it', 'those', 'these']
        return any(keyword in topic.lower() for keyword in ambiguous_keywords)

    def get_follow_up_question(self, topic):
        """
        Generate a follow-up question to clarify the ambiguous topic.
        """
        return f"Could you please clarify what you mean by '{topic}'?"

    def create_prompt(self, topic, conversation_history):
        """
        Create a prompt for generating an outline, incorporating the topic and conversation history.
        """
        return (
            f"I'm tasked with creating an educational guide for prospective students interested in {topic} at Western Sydney University International College. "
            f"Based on our previous discussions, what key sections and topics should I include to ensure the guide is comprehensive? "
            f"Here's our conversation so far:\n{conversation_history}"
        )

    def format_history(self, conversation_history):
        """
        Format the conversation history into a readable format for the LLM prompt.
        """
        if not conversation_history:
            return "No prior conversation history."
        
        formatted_history = "\n".join([f"{entry['role'].capitalize()}: {entry['content']}" for entry in conversation_history])
        return formatted_history.strip()

    @staticmethod
    def sanitize_input(text):
        # Strip only leading and trailing whitespace to preserve internal spaces
        return text.strip()

# Example usage
outline_agent = OutlineAgent()
