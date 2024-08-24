import threading
import tkinter as tk
from tkinter import scrolledtext
from planner_agent import PlannerAgent
import logging

# Configure logging
logging.basicConfig(filename='chatbot_activity.log', level=logging.INFO, format='%(asctime)s:%(levelname)s:%(message)s')

# Initialize PlannerAgent
planner_agent = PlannerAgent()

def sanitize_input(text):
    return text.strip()  # Remove leading and trailing whitespaces

def generate_response(text):
    try:
        sanitized_text = sanitize_input(text)
        logging.info(f"Processing sanitized input: {sanitized_text}")
        response = planner_agent.plan_and_execute(sanitized_text)
        logging.info(f"Generated response: {response}")
        return response
    except Exception as e:
        error_message = f"An error occurred: {str(e)}"
        logging.error(error_message)
        return error_message

def update_chat_history(user_input, response):
    chat_history.config(state=tk.NORMAL)
    chat_history.insert(tk.END, f"You: {user_input}\nBot: {response}\n")
    chat_history.yview(tk.END)
    chat_history.config(state=tk.DISABLED)

def handle_query():
    user_input = entry_field.get("1.0", tk.END).strip()
    if not user_input:  # Ensure the input is not empty
        return
    
    logging.info(f"User input: {user_input}")
    entry_field.config(state=tk.DISABLED)  # Disable entry to prevent multiple sends
    send_button.config(state=tk.DISABLED)  # Disable the send button

    # Process in a separate thread to keep the UI responsive
    def process_input():
        response = generate_response(user_input)
        root.after(0, update_chat_history, user_input, response)
        root.after(0, reset_input)

    threading.Thread(target=process_input).start()

def reset_input():
    entry_field.config(state=tk.NORMAL)  # Re-enable entry after response
    entry_field.delete("1.0", tk.END)
    send_button.config(state=tk.NORMAL)  # Re-enable the send button

# Set up the main application window
root = tk.Tk()
root.title("WSU Chat")
root.configure(bg="#282C34")

chat_history = scrolledtext.ScrolledText(root, state='disabled', width=60, height=20, bg="#1E1E1E", fg="#61AFEF", font=("Helvetica", 14), wrap="word", padx=10, pady=10)
chat_history.grid(row=0, column=0, columnspan=2, padx=10, pady=10)

entry_field = scrolledtext.ScrolledText(root, height=3, width=50, bg="#1E1E1E", fg="#ABB2BF", font=("Helvetica", 14), wrap="word", padx=10, pady=10)
entry_field.grid(row=1, column=0, padx=10, pady=10)

send_button = tk.Button(root, text="Send", command=handle_query, bg="#61AFEF", fg="#282C34", font=("Helvetica", 14, "bold"), padx=20, pady=10, activebackground="#98C379")
send_button.grid(row=1, column=1, padx=10, pady=10)

root.mainloop()
