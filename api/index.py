from flask import Flask, jsonify

app = Flask(__name__)

@app.route("/")
def home():
    return jsonify(message="Hello from Vercel!")

# Vercel needs this as the entry point
app = app
