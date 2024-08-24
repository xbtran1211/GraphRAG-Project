import logging
from huggingface_hub import InferenceClient
import re

class GenerativeAgent:
    def __init__(self, model_id="meta-llama/Meta-Llama-3-8B-Instruct", token="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"):
        self.client = InferenceClient(model=model_id, token=token)
        self.max_recursion_depth = 3  # Limit recursive calls to prevent infinite loops
        self.max_tokens = 300  # Limit the number of tokens to prevent overly long responses

    def generate_response(self, input_text, context_history):
        """
        Generate a response based on the input text and the conversation context.
        :param input_text: The new user input
        :param context_history: A list of dictionaries representing the conversation history
        """
        try:
            # Construct the conversation history into a single prompt for the LLM
            formatted_history = self.format_history(context_history)
            prompt = self.create_focused_prompt(input_text, formatted_history)

            logging.info(f"Sending request to generate response with context. Input: '{input_text}'")

            # Make the initial API call with the conversation context
            generated_text = self.get_full_response(prompt, recursion_depth=0)

            logging.info(f"Final generated text: {generated_text}")
            return generated_text
        except Exception as e:
            error_msg = f"An error occurred: {str(e)}"
            logging.error(error_msg)
            return "Sorry, I encountered an unexpected issue. Please try again later."

    def get_full_response(self, prompt, recursion_depth):
        """
        Recursively fetches the complete response from the LLM by checking for sentence and content completion.
        """
        if recursion_depth > self.max_recursion_depth:
            return "Sorry, I'm having trouble completing my response."

        response = self.client.chat_completion(
            messages=[{"role": "system", "content": prompt}],
            max_tokens=self.max_tokens  # Adjust token limit to control response length
        )

        if response and response.choices and len(response.choices) > 0:
            generated_text = response.choices[0].message.content.strip()

            # Check if the response is complete
            if self.response_is_incomplete(generated_text):
                logging.info("Response appears to be incomplete, querying for more data...")
                more_text = self.get_full_response(generated_text, recursion_depth + 1)
                generated_text += " " + more_text.strip()

            return generated_text
        else:
            error_msg = "No valid response received from the API."
            logging.error(error_msg)
            return "I'm having trouble accessing my response capabilities at the moment. Could you please try rephrasing your question or ask something else?"

    def response_is_incomplete(self, text):
        """
        Check if the response ends with a proper sentence-ending punctuation and whether it sufficiently addresses the query.
        """
        # Check for grammatical completeness
        if not text.endswith(('.', '!', '?')):
            return True

        # Add checks for content sufficiency (you can expand this to check for keywords, length, etc.)
        if len(text.split()) < 50:  # Example: Minimum word count for a complete response
            return True

        return False

    def create_focused_prompt(self, input_text, formatted_history):
        """
        Create a focused prompt that guides the LLM to stay on track with the query.
        """
        return (
            f"You are a helpful assistant. Please provide a clear, concise response to the user's inquiry.\n"
            f"User's question: '{input_text}'.\n"
            f"Conversation history:\n{formatted_history}\n"
            "Limit your response to relevant information and avoid unnecessary elaboration."
        )

    def format_history(self, context_history):
        """
        Formats the conversation history into a readable format for the LLM prompt.
        :param context_history: A list of dictionaries, each containing 'role' and 'content' keys
        :return: A formatted string representing the conversation history
        """
        formatted_history = ""
        for entry in context_history:
            role = entry["role"]
            content = entry["content"]
            formatted_history += f"{role.capitalize()}: {content}\n"
        return formatted_history.strip()
