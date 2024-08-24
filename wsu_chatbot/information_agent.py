import logging
import spacy
from utils import query_llm_api
from language_utils import LanguageDetector

# Configure logging
logging.basicConfig(level=logging.INFO)

class InformationAgent:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")  # Load the pre-trained spaCy model for NER
        self.DEFAULT_LANGUAGE = 'en_XX'
        self.language_detector = LanguageDetector()  # Assuming it's initialized properly

    def generate_detailed_answer(self, topic, conversation_history, gathered_info):
        # Enhance entity extraction by considering all text inputs
        entities = self.extract_entities(topic + " " + " ".join([h['content'] for h in conversation_history]))
        logging.info(f"Extracted entities from topic and conversation history: {entities}")

        # Improve language detection by considering all relevant text
        detected_lang = self.language_detector.detect_language_ft(" ".join([topic, gathered_info] + [h['content'] for h in conversation_history])) or self.DEFAULT_LANGUAGE

        # Build a concise prompt with extracted entities and conversation history
        prompt = self.create_prompt(topic, conversation_history, gathered_info, entities)
        logging.info(f"Sending enhanced prompt to LLM: {prompt}")
        
        try:
            response = self.get_limited_response(prompt, detected_lang)
            if response:
                logging.info(f"Received response from LLM: {response}")
                return self.format_response(response)
            else:
                raise ValueError("Empty or invalid response received from the API.")
        except Exception as e:
            error_msg = f"Error querying LLM: {str(e)}"
            logging.error(error_msg)
            return f"An error occurred: {error_msg}"

    def extract_entities(self, text):
        # Perform NER on the extended text, ignoring irrelevant entities
        doc = self.nlp(text)
        return [(ent.text, ent.label_) for ent in doc.ents if ent.text.lower() != "generative"]

    def create_prompt(self, topic, conversation_history, gathered_info, entities):
        # Build a concise prompt using formatted history and extracted entities
        entities_str = ", ".join([f"{entity[0]} ({entity[1]})" for entity in entities]) or "No significant entities detected"
        formatted_history = self.format_history(conversation_history)
        
        return (
            f"You are an experienced information officer at Western Sydney University. "
            f"Your task is to respond concisely to a student's inquiry on the topic '{topic}'. "
            f"Entities detected: {entities_str}. "
            f"Conversation history:\n{formatted_history}\n"
            f"Use the gathered information: '{gathered_info}' to craft a brief and clear response."
        )

    def get_limited_response(self, prompt, detected_lang):
        """
        Fetches a concise response from the LLM, limiting verbosity.
        """
        response = query_llm_api(prompt, detected_lang)

        # Limit the response to a maximum length of 200 words
        if response and len(response.split()) > 200:
            response = " ".join(response.split()[:200]) + "..."
        return response

    def is_complete_sentence(self, text):
        """
        Check if the text ends with proper sentence-ending punctuation.
        """
        return text.endswith(('.', '!', '?'))

    def format_history(self, conversation_history):
        # Format the conversation history into a readable string for the prompt
        return "\n".join([f"{entry['role'].capitalize()}: {entry['content']}" for entry in conversation_history])

    def format_response(self, response):
        # Apply better formatting to the LLM response for readability
        formatted_response = f"Here’s what we found:\n\n{response}"
        
        # Improve readability with formatting, like bullet points or paragraphs
        formatted_response = formatted_response.replace("1.", "\n1.").replace("2.", "\n2.").replace("3.", "\n3.")
        return formatted_response
