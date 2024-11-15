from flask import Flask, request, jsonify
import openai

app = Flask(__name__)

# Set your OpenAI API key
openai.api_key = "your_openai_api_key"

@app.route("/", methods=["POST"])
def chat():
    try:
        data = request.json
        user_message = data.get("message", "")
        
        # Call the OpenAI ChatGPT API
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": user_message}]
        )

        # Extract response text
        chat_response = response['choices'][0]['message']['content']
        return jsonify({"reply": chat_response})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Entry point for Vercel
app = app
