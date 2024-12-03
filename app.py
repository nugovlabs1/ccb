import os
import json
import openai
import numpy as np
import faiss
from flask import Flask, request, jsonify
from flask_cors import CORS
import time

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Set OpenAI API key
openai.api_key = "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"

# Paths for FAISS index and metadata
current_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(current_dir, "../outputs")
index_file = os.path.join(output_dir, "chunk_index.faiss")
metadata_file = os.path.join(output_dir, "chunk_embeddings.json")

# Load FAISS index and chunk metadata
print("Loading FAISS index and metadata...")
index = faiss.read_index(index_file)
with open(metadata_file, "r") as f:
    chunk_metadata = json.load(f)
print("Loaded FAISS index and metadata.")

# Function to retrieve relevant chunks based on query
def get_relevant_chunks(query, top_k=3):
    try:
        print("Generating query embedding...")
        response = openai.Embedding.create(
            model="text-embedding-ada-002",
            input=query
        )
        query_embedding = np.array(response["data"][0]["embedding"], dtype=np.float32).reshape(1, -1)

        print("Searching FAISS index for relevant chunks...")
        distances, indices = index.search(query_embedding, top_k)

        relevant_chunks = []
        for idx in indices[0]:
            relevant_chunks.append(chunk_metadata[idx]["text"])
        print("Retrieved relevant chunks.")
        return relevant_chunks
    except Exception as e:
        print(f"Error retrieving relevant chunks: {e}")
        return ["An error occurred while retrieving relevant information. Please try again later."]

# GPT-4 Chat Function with Retry Logic
def query_gpt4(question, context, max_retries=3, delay=5):
    print("Querying GPT-4...")
    retries = 0
    while retries < max_retries:
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant for city-related queries."},
                    {"role": "user", "content": f"Context: {context}\n\nQuestion: {question}"}
                ]
            )
            return response['choices'][0]['message']['content']
        except openai.error.APIError as e:
            print(f"API Error: {e}")
            retries += 1
            if retries >= max_retries:
                raise
            print(f"Retrying in {delay} seconds...")
            time.sleep(delay)
        except Exception as e:
            print(f"Unexpected error querying GPT-4: {e}")
            return "An unexpected error occurred while generating the response. Please try again later."

# API endpoint for chatbot interaction
@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    question = data.get("question", "")

    if not question:
        return jsonify({"error": "No question provided."}), 400

    try:
        # Retrieve relevant chunks and combine them into a single context
        relevant_chunks = get_relevant_chunks(question, top_k=3)
        context = "\n\n".join(relevant_chunks)

        # Query GPT-4 with the relevant context
        answer = query_gpt4(question, context)
        return jsonify({"answer": answer})
    except openai.error.APIError as e:
        print(f"OpenAI API Error: {e}")
        return jsonify({"error": "The server is temporarily unavailable. Please try again later."}), 503
    except Exception as e:
        print(f"Unexpected Error: {e}")
        return jsonify({"error": "An unexpected error occurred."}), 500

# Serve the frontend (index.html)
@app.route("/")
def serve_index():
    try:
        return open(os.path.join(current_dir, "index.html")).read()
    except Exception as e:
        print(f"Error serving index.html: {e}")
        return "Error loading the homepage. Please check the server configuration.", 500

# Run Flask app
if __name__ == "__main__":
    app.run(debug=True)
