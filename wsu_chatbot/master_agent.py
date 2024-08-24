import logging

class MasterAgent:
    def __init__(self):
        from topic_agent import TopicAgent
        from perspective_agent import PerspectiveAgent
        from questioning_agent import QuestioningAgent
        from information_agent import InformationAgent
        from outline_agent import OutlineAgent
        from generative_agent import GenerativeAgent
        from intent_classifier import IntentClassifier

        # Initialize agents
        self.topic_agent = TopicAgent()
        self.perspective_agent = PerspectiveAgent()
        self.questioning_agent = QuestioningAgent()
        self.information_agent = InformationAgent()
        self.outline_agent = OutlineAgent()
        self.generative_agent = GenerativeAgent()
        self.intent_classifier = IntentClassifier()

        # Store conversation history
        self.context_history = []

    def update_context(self, user_query, bot_response):
        """
        Update context history by appending the latest user and bot responses.
        Limit the history to the last 10 exchanges to prevent overflow.
        """
        self.context_history.append({"role": "user", "content": user_query})
        self.context_history.append({"role": "bot", "content": bot_response})
        if len(self.context_history) > 10:
            self.context_history = self.context_history[-10:]

    def process_query(self, user_query, tasks=None):
        """
        Process the user query by determining and executing the relevant tasks.
        """
        # Determine tasks if not explicitly provided
        tasks = tasks or self.determine_tasks(user_query)
        responses = {}

        if not tasks:
            # Handle intent-based fallback
            return self.handle_intent_responses(user_query)

        # Execute tasks based on their handlers
        for task in tasks:
            method = getattr(self, f'execute_{task}', self.execute_default)
            if task == 'generative':
                # Ensure generative responses handle incomplete text
                responses[task] = self.generative_agent.handle_incomplete_response(user_query, self.context_history)
            else:
                responses[task] = method(user_query, 'en_XX', self.context_history)

        # Compile the final response from the results of the task execution
        final_response = self.compile_final_response(responses)
        self.update_context(user_query, final_response)
        return final_response

    def compile_final_response(self, responses):
        """
        Combine all the responses from different tasks into a final string.
        """
        return "\n\n".join(f"{key.replace('_', ' ').title()}:\n{value}" for key, value in responses.items())

    def handle_intent_responses(self, user_query):
        """
        Determine the user's intent using the intent classifier and respond accordingly.
        """
        intent = self.intent_classifier.classify_intent(user_query)
        logging.info(f"Classified intent: {intent}")

        if intent == "greeting":
            return {'greeting': self.handle_greeting()}
        elif intent == "negative response":
            return {'negative_response': self.handle_negative_response()}
        elif intent == "information request":
            # Route the query to the information agent
            return self.process_query(user_query, tasks=['information'])
        else:
            # Fallback to generative response for other intents
            return {'default': self.execute_generative(user_query, self.context_history)}

    def execute_information(self, user_query, language, context_history):
        """
        Use the InformationAgent to generate a detailed response based on the user's query.
        """
        return self.information_agent.generate_detailed_answer(user_query, context_history, "")

    def execute_generative(self, user_query, context_history):
        """
        Generate a response using the GenerativeAgent, ensuring handling of multi-part responses.
        """
        return self.generative_agent.handle_incomplete_response(user_query, context_history)

    def execute_default(self, user_query, language, context_history=None):
        """
        Default handler for tasks that don't have a specific execution method.
        """
        logging.warning(f"No specific handler for task. Input was: {user_query}")
        return f"I'm not sure how to handle this request: '{user_query}'. Could you please clarify?"

    def handle_greeting(self):
        """
        Simple greeting response for basic interactions.
        """
        return "Hello! How can I assist you today?"

    def handle_negative_response(self):
        """
        Respond to negative sentiment from the user.
        """
        return "I understand. If you have any other questions or need assistance with something else, feel free to ask!"

    def determine_tasks(self, query):
        """
        Determine which tasks to perform based on the query content.
        If no specific tasks match, fallback to generative response.
        """
        tasks = []
        query_lower = query.lower()

        # Add specific tasks based on keywords in the query
        if "wsu" in query_lower:
            tasks.append('information')
        elif any(kw in query_lower for kw in ["help", "support"]):
            tasks.append('support')

        # Default to generative response if no tasks are determined
        if not tasks:
            tasks.append('generative')

        return tasks
