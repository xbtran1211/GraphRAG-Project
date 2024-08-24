import logging
from huggingface_hub import InferenceClient

# Configure logging
logging.basicConfig(level=logging.INFO)

class IntentClassifier:
    def __init__(self, model_id="meta-llama/Meta-Llama-3-8B-Instruct", token="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"):
        self.client = InferenceClient(model=model_id, token=token)
        self.common_intents_cache = {}  # Cache to store common intents
        self.DEFAULT_INTENT = "clarification request"  # Default fallback intent

    def classify_intent(self, user_query, conversation_history=None):
        """
        Classify the intent of the user's query using the conversation history if available.
        """
        # Use conversation history to help in intent classification
        if conversation_history:
            formatted_history = self.format_history(conversation_history)
            full_prompt = (
                f"Based on the following conversation: '{formatted_history}', classify the intent of the user's most recent query: '{user_query}'. "
                f"The possible intents are: greeting, negative response, information request, goodbye, clarification request, etc. "
                f"Answer with the single most likely intent."
            )
        else:
            # Fallback to the query itself if no conversation history is provided
            full_prompt = (
                f"Classify the intent of this query: '{user_query}'. "
                f"The possible intents are: greeting, negative response, information request, goodbye, clarification request, etc. "
                f"Answer with the single most likely intent."
            )

        # Check if the intent is already cached
        if user_query in self.common_intents_cache:
            logging.info(f"Intent found in cache for query: {user_query}")
            return self.common_intents_cache[user_query]

        try:
            logging.info(f"Classifying intent for query: {user_query}")
            response = self.client.chat_completion(
                messages=[{"role": "system", "content": full_prompt}],
                max_tokens=50  # Reduced token limit for efficiency
            )

            # Check for response validity
            if response and response.choices:
                generated_text = response.choices[0].message.content.lower().strip()

                # Log the generated response
                logging.info(f"Generated response for intent classification: {generated_text}")

                # Cache the result if it's a common query
                if user_query in ["hello", "hi", "goodbye", "thanks"]:
                    self.common_intents_cache[user_query] = generated_text

                return generated_text
            else:
                logging.error(f"Invalid response received: {response}")
                return self.DEFAULT_INTENT

        except Exception as e:
            logging.error(f"Error in intent classification using LLM: {str(e)}")
            return self.DEFAULT_INTENT

    def format_history(self, conversation_history):
        """
        Format the conversation history into a readable format for the LLM prompt.
        """
        formatted_history = ""
        for entry in conversation_history:
            role = entry["role"]
            content = entry["content"]
            formatted_history += f"{role.capitalize()}: {content}\n"
        return formatted_history.strip()

    def generate_response(self, prompt):
        """
        Generate a response from the LLM based on the prompt.
        """
        try:
            logging.info(f"Generating response for prompt: {prompt}")
            response = self.client.chat_completion(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=300  # Adjust token count as needed
            )

            # Check for response validity
            if response and response.choices:
                generated_text = response.choices[0].message.content.strip()
                logging.info(f"Generated response: {generated_text}")
                return generated_text
            else:
                logging.error(f"Invalid response received: {response}")
                return "Unable to generate response."

        except Exception as e:
            logging.error(f"Error generating response: {str(e)}")
            return f"An error occurred while generating the response: {str(e)}"

# Example usage
intent_classifier = IntentClassifier()
