import openai
import os
import json
import faiss
import numpy as np

# Set your OpenAI API key
openai.api_key = "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"

# Define paths
current_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(current_dir, "../outputs")
embeddings_file = os.path.join(current_dir, "../outputs/chunk_embeddings.json")

# Initialize FAISS index
dimension = 1536  # Embedding dimension for text-embedding-ada-002
index = faiss.IndexFlatL2(dimension)
chunk_metadata = []

# Function to split text into smaller chunks
def split_text_into_chunks(text, max_tokens=8000):
    words = text.split()
    chunks = []
    current_chunk = []
    current_length = 0

    for word in words:
        word_length = len(word) + 1  # Include space
        if current_length + word_length > max_tokens:
            chunks.append(" ".join(current_chunk))
            current_chunk = []
            current_length = 0
        current_chunk.append(word)
        current_length += word_length

    if current_chunk:
        chunks.append(" ".join(current_chunk))
    return chunks

# Generate embeddings for each chunk
for file_name in sorted(os.listdir(output_dir)):
    if file_name.endswith(".txt"):
        with open(os.path.join(output_dir, file_name), "r") as f:
            chunk_text = f.read()

        # Split the text into smaller sub-chunks
        sub_chunks = split_text_into_chunks(chunk_text, max_tokens=8000)

        for i, sub_chunk in enumerate(sub_chunks):
            # Generate embedding for each sub-chunk
            response = openai.Embedding.create(
                model="text-embedding-ada-002",
                input=sub_chunk
            )
            embedding = np.array(response["data"][0]["embedding"], dtype=np.float32)

            # Add embedding to FAISS index
            index.add(np.array([embedding]))
            chunk_metadata.append({
                "file_name": file_name,
                "sub_chunk_index": i,
                "text": sub_chunk
            })

# Save FAISS index and metadata
faiss.write_index(index, os.path.join(current_dir, "../outputs/chunk_index.faiss"))
with open(embeddings_file, "w") as f:
    json.dump(chunk_metadata, f)

print("Embeddings generated and saved.")
