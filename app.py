from flask import Flask, render_template, request, jsonify
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import datetime
import json
import re
from googlesearch import search
import mysql.connector

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")

# Set pad token for handling attention mask
tokenizer.pad_token = tokenizer.eos_token

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Initialize Flask app
app = Flask(__name__)

# Store chat history for multiple users
chat_histories = {}

# Connect to MySQL database
def get_db_connection():
    return mysql.connector.connect(
        host="localhost",
        user="root",  # Default user for XAMPP
        password="",  # Default password is empty
        database="chatbot_db"
    )

# Save chat history to the "information" table
def save_chat_to_db(user_request, bot_response):
    conn = get_db_connection()
    cursor = conn.cursor()

    query = "INSERT INTO information (request, response) VALUES (%s, %s)"
    cursor.execute(query, (user_request, bot_response))

    conn.commit()
    cursor.close()
    conn.close()

# Fetch chat history from the "information" table
@app.route("/history", methods=["GET"])
def get_chat_history():
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)

    query = "SELECT request, response, timestamp FROM information ORDER BY timestamp DESC"
    cursor.execute(query)
    history = cursor.fetchall()

    cursor.close()
    conn.close()
    
    return jsonify(history)

# Load predefined responses from JSON
def load_json_data():
    try:
        with open('intents.json', 'r') as file:
            data = json.load(file)
        return data
    except FileNotFoundError:
        print("JSON file not found.")
        return {}

response_data = load_json_data()

# Generate chatbot response using DialoGPT
def get_chat_response(text, user_id):
    if user_id not in chat_histories:
        chat_histories[user_id] = None

    new_user_input_ids = tokenizer.encode(text + tokenizer.eos_token, return_tensors='pt').to(device)
    attention_mask = torch.ones_like(new_user_input_ids)

    if chat_histories[user_id] is not None:
        bot_input_ids = torch.cat([chat_histories[user_id], new_user_input_ids], dim=-1)
        attention_mask = torch.cat([torch.ones_like(chat_histories[user_id]), attention_mask], dim=-1)
    else:
        bot_input_ids = new_user_input_ids

    chat_histories[user_id] = model.generate(
        bot_input_ids,
        max_length=1000,
        pad_token_id=tokenizer.eos_token_id,
        temperature=0.8,
        top_k=40,
        top_p=0.85,
        do_sample=True,
        repetition_penalty=1.15,
        attention_mask=attention_mask
    )
    
    response = tokenizer.decode(chat_histories[user_id][:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)
    return response

# Greetings based on time
def wish():
    hour = datetime.datetime.now().hour
    if hour < 12:
        return "Good Morning! How may I assist you?"
    elif hour < 18:
        return "Good Afternoon! How may I assist you?"
    else:
        return "Good Evening! How may I assist you?"

# Google search
def google_search(query):
    try:
        results = list(search(query, num=3))
        if results:
            return f"Here are the top search results:\n1. {results[0]}\n2. {results[1]}\n3. {results[2]}"
        else:
            return "Sorry, I couldn't find any results."
    except Exception as e:
        return f"An error occurred while searching: {e}"

# Get current time
def get_time():
    return f"The current time is {datetime.datetime.now().strftime('%I:%M %p')}"

# Get today's date
def get_date():
    return f"Today's date is {datetime.date.today().strftime('%B %d, %Y')}"

# Search JSON responses
def search_json_response(query):
    for key, value in response_data.items():
        if key.lower() in query.lower():
            return value
    return None

# Math operations
def handle_math_operations(user_input):
    math_pattern = re.compile(r"(\d+)\s*(\+|\-|\*|\/)\s*(\d+)")
    match = math_pattern.search(user_input)
    
    if match:
        num1 = float(match.group(1))
        operator = match.group(2)
        num2 = float(match.group(3))
        
        if operator == "+":
            result = num1 + num2
        elif operator == "-":
            result = num1 - num2
        elif operator == "*":
            result = num1 * num2
        elif operator == "/":
            if num2 == 0:
                return "Division by zero is not allowed!"
            result = num1 / num2
        
        return f"The result of {num1} {operator} {num2} is {result}."
    else:
        return None

# Generate response
def respond(user_input):
    user_input = user_input.lower()
    
    # Check for predefined responses
    json_response = search_json_response(user_input)
    if json_response:
        return json_response
    
    # Check for math operations
    math_response = handle_math_operations(user_input)
    if math_response:
        return math_response

    # Handle basic queries
    if "time" in user_input:
        return get_time()
    elif "date" in user_input:
        return get_date()
    elif "hello" in user_input or "hi" in user_input:
        return wish()
    elif "search" in user_input:
        query = user_input.replace("search", "").strip()
        return google_search(query)
    elif "exit" in user_input or "bye" in user_input:
        return "Goodbye! Have a great day."
    else:
        return get_chat_response(user_input, user_id="general_user")

# Main page
@app.route("/")
def index():
    return render_template('chat.html')

# Chat route
@app.route("/get", methods=["POST"])
def chat():
    user_input = request.form["msg"].strip()
    response = respond(user_input)

    # Save to "information" table
    save_chat_to_db(user_input, response)

    return jsonify({"response": response})

# Run Flask app
if __name__ == '__main__':
    app.run(debug=True)
