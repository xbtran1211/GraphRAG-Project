import logging
import spacy
from utils import query_llm_api
from language_utils import LanguageDetector  # Ensure this is implemented properly

class QuestioningAgent:
    def __init__(self, model_path='C:/Users/manus/fastText/lid.176.bin'):
        self.language_detector = LanguageDetector(model_path)
        self.DEFAULT_LANGUAGE = 'en_XX'
        self.nlp = spacy.load("en_core_web_sm")  # Use spaCy for deeper analysis

    def generate_follow_up_questions(self, topic, conversation_history):
        """
        Generates follow-up questions based on the topic and conversation history, with ambiguity handling.
        """
        # Sanitize inputs
        sanitized_topic = topic.strip()

        # Handle empty topic cases
        if not sanitized_topic:
            logging.warning("Empty topic provided.")
            return "Can you please provide more information about the topic?"

        # Format conversation history for the prompt
        formatted_history = self.format_history(conversation_history)

        # Create the prompt using the sanitized topic and formatted conversation history
        prompt = self.create_prompt(sanitized_topic, formatted_history)

        # Detect language for better contextual alignment
        detected_lang = self.language_detector.detect_language_ft(formatted_history) or self.DEFAULT_LANGUAGE

        # Detect if there is ambiguity
        if self.is_ambiguous(sanitized_topic, formatted_history):
            follow_up_question = self.get_follow_up_question(sanitized_topic)
            logging.info(f"Generated follow-up question to clarify ambiguity: {follow_up_question}")
            return follow_up_question

        try:
            logging.info(f"Generating follow-up questions for topic '{sanitized_topic}' in language: {detected_lang}")
            response = query_llm_api(prompt, detected_lang)
            logging.info(f"Received follow-up questions: {response}")
            return response
        except Exception as e:
            logging.error(f"Error generating follow-up questions for topic '{sanitized_topic}': {str(e)}")
            return f"An error occurred while generating follow-up questions: {str(e)}"

    def format_history(self, conversation_history):
        """
        Format the conversation history for use in the LLM prompt.
        """
        formatted_history = ""
        for entry in conversation_history:
            role = entry["role"]
            content = entry["content"]
            formatted_history += f"{role.capitalize()}: {content}\n"
        return formatted_history.strip()

    def is_ambiguous(self, topic, conversation_history):
        """
        Check if the input is ambiguous based on the topic and conversation history.
        Use both keyword and NLP-based analysis for better results.
        """
        ambiguous_keywords = ['this', 'that', 'there', 'it', 'these', 'those']

        # Keyword check
        if any(keyword in topic for keyword in ambiguous_keywords):
            return True

        # NLP-based analysis: Check for incomplete or vague sentences
        doc = self.nlp(topic)
        if len(doc) < 3 or not any(token.dep_ == 'ROOT' for token in doc):
            return True

        return False

    def get_follow_up_question(self, topic):
        """
        Generate a clarifying follow-up question when ambiguity is detected.
        """
        return f"Could you please clarify what you mean by '{topic}'?"

    def create_prompt(self, topic, conversation_history):
        """
        Create a prompt for generating follow-up questions, incorporating the topic and conversation history.
        """
        return (
            f"You are an experienced study consultant with a focus on conducting thorough research on the topic '{topic}'. "
            f"You are currently engaging in a conversation with an expert to gather deeper insights. "
            f"Consider the previous discussions: '{conversation_history}', and ask insightful, topic-specific questions "
            f"to uncover the most valuable information. Avoid repeating any questions from earlier discussions and ensure each new question "
            f"deepens your understanding of the topic. Once you feel the information gathered is sufficient, "
            f"politely conclude the conversation by saying, 'Thank you so much for your help!' "
            f"Remember to ask only one question at a time."
        )
