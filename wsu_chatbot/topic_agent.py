import logging
from utils import query_llm_api, detect_language

# Configure logging
logging.basicConfig(level=logging.INFO)

class TopicAgent:
    DEFAULT_LANGUAGE = "en_XX"  # Default to English if language detection fails

    def __init__(self):
        self.context_history = []  # To store the conversation history

    def generate_related_topics(self, topic):
        # Update conversation history
        self.update_context(topic)

        # Sanitize the input topic
        sanitized_topic = topic.strip()
        formatted_history = self.format_history()

        # Create the prompt using the sanitized topic and conversation history
        prompt = self.create_prompt(sanitized_topic, formatted_history)
        
        # Detect the language of the topic or use the default language
        detected_lang = detect_language(sanitized_topic) or self.DEFAULT_LANGUAGE
        
        # Handle ambiguity in the topic
        if self.is_ambiguous(sanitized_topic):
            follow_up_question = self.get_follow_up_question(sanitized_topic)
            logging.info(f"Generated follow-up question to clarify ambiguity: {follow_up_question}")
            return follow_up_question

        try:
            # Log the action of sending a prompt
            logging.info(f"Sending prompt: {prompt} in language: {detected_lang}")
            
            # Query the language model API and log the response
            response = query_llm_api(prompt, detected_lang)
            logging.info(f"Received response: {response}")
            
            return response
        except Exception as e:
            # Log and return the error
            logging.error(f"Error in generating topics for topic '{sanitized_topic}': {str(e)}")
            return f"An error occurred while generating related topics: {str(e)}"

    def is_ambiguous(self, topic):
        """
        Enhanced check for ambiguity by using spaCy's NLP model to identify vague or incomplete topics.
        """
        # Simple keyword-based check
        ambiguous_keywords = ['this', 'that', 'it', 'those', 'these']
    
        # Check if topic contains any ambiguous keywords
        if any(keyword in topic for keyword in ambiguous_keywords):
            return True
    
        # Additional ambiguity check using NLP techniques (e.g., identifying incomplete or vague statements)
        doc = self.nlp(topic)
        if len(doc) < 3:  # Consider very short inputs as potentially ambiguous
            return True
    
        # More complex rules can be added here
        return False

    def get_follow_up_question(self, topic):
        """
        Generate a follow-up question to clarify the ambiguous topic.
        """
        return f"Could you please clarify what you mean by '{topic}'?"

    def create_prompt(self, topic, conversation_history):
        """
        Create a prompt that incorporates the topic and the conversation history for generating related topics.
        """
        return (
            f"I am a prospective student or parent interested in learning more about '{topic}'. "
            f"Based on our previous discussions, could you provide examples of related topics that offer insights into key aspects commonly associated with this topic? "
            f"I am also interested in understanding the typical content and structure of similar topics, so examples of related topics that could help with that would be appreciated."
            f"\nHere is the conversation history:\n{conversation_history}"
        )

    def update_context(self, topic):
        """
        Update the context history with the latest topic.
        """
        self.context_history.append({"role": "user", "content": topic})
        if len(self.context_history) > 5:  # Limit history to the last 5 entries
            self.context_history.pop(0)

    def format_history(self):
        """
        Format the conversation history for inclusion in the prompt.
        """
        formatted_history = ""
        for entry in self.context_history:
            role = entry["role"]
            content = entry["content"]
            formatted_history += f"{role.capitalize()}: {content}\n"
        return formatted_history.strip()

# Example usage
topic_agent = TopicAgent()
