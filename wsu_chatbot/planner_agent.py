import logging
from master_agent import MasterAgent
from language_utils import LanguageDetector

# Set up logging
logging.basicConfig(level=logging.INFO)

class PlannerAgent:
    def __init__(self):
        self.master_agent = MasterAgent()
        self.language_detector = LanguageDetector()
        self.language_detector.load_model()  # Ensure the language model is loaded
        self.conversation_history = []

    def plan_and_execute(self, user_query):
        logging.info(f"Received user query: {user_query}")
        self.conversation_history.append({"role": "user", "content": user_query})

        # Detect and log the language of the query
        language = self.detect_language(user_query)

        # Normalize and log the query
        normalized_query = self.pre_process_query(user_query)

        # Determine and log tasks
        tasks = self.determine_and_prioritize_tasks(normalized_query)

        # Attempt to execute tasks and handle any errors
        try:
            responses = self.master_agent.process_query(normalized_query, tasks)
            logging.info(f"Responses received from MasterAgent: {responses}")
        except Exception as e:
            error_message = f"Error processing query: {str(e)}"
            logging.error(error_message)
            return self.handle_error(error_message)

        # Format and return the final response
        final_response = self.post_process_response(responses)
        self.conversation_history.append({"role": "bot", "content": final_response})

        return final_response

    def detect_language(self, query):
        language = self.language_detector.detect_language_ft(query)
        if not language:
            language = 'en_XX'  # Fallback to English if detection fails
        logging.info(f"Detected language: {language}")
        return language

    def pre_process_query(self, query):
        # Normalize the query by trimming whitespace and converting to lowercase
        normalized_query = query.strip().lower()
        logging.info(f"Normalized query: {normalized_query}")
        return normalized_query

    def determine_and_prioritize_tasks(self, query):
        # Determine the tasks using the MasterAgent
        tasks = self.master_agent.determine_tasks(query)

        # Prioritize the 'information' task if it exists
        if 'information' in tasks:
            tasks.remove('information')
            tasks.insert(0, 'information')

        logging.info(f"Determined tasks: {tasks}")
        return tasks

    def post_process_response(self, responses):
        # Handle cases where no responses are returned
        if not responses:
            logging.warning("No responses were generated.")
            return "Is there something specific you'd like to know about? I'm here to help."

        # Handle case where responses is a string (fallback or error)
        if isinstance(responses, str):
            logging.info("Single string response detected.")
            return f"Here’s what we found:\n\nWSU CHAT:\n{responses}"

        # Compile the final response from the responses dictionary
        try:
            final_response_parts = []
            for key, value in responses.items():
                if key == 'generative':
                    final_response_parts.append(f"WSU CHAT:\n{value}")
                else:
                    final_response_parts.append(f"**{key.replace('_', ' ').title()}**:\n{value}")

            final_response = "Here’s what we found:\n\n" + "\n\n".join(final_response_parts)
            logging.info("Compiled final response")
            return final_response.strip()

        except AttributeError as e:
            logging.error(f"Error compiling final response: {str(e)}")
            return "An error occurred while compiling the response. Please try again."

    def handle_error(self, error_message):
        # Handle errors by logging and returning a user-friendly message
        logging.error(f"Handling error: {error_message}")
        return f"An error occurred: {error_message}"

    def format_history(self):
        # Format the conversation history for use by the MasterAgent or other agents
        formatted_history = ""
        for entry in self.conversation_history:
            role = entry["role"]
            content = entry["content"]
            formatted_history += f"{role.capitalize()}: {content}\n"
        return formatted_history.strip()
