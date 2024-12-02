import os
import re

# Define paths
current_dir = os.path.dirname(os.path.abspath(__file__))
input_dir = os.path.join(current_dir, "../processed_data")
output_dir = os.path.join(current_dir, "../outputs")

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

# Function to clean and chunk text
def clean_text(text):
    return re.sub(r'\s+', ' ', text).strip()

def chunk_text(text, chunk_size=7500):
    words = text.split()
    for i in range(0, len(words), chunk_size):
        yield ' '.join(words[i:i + chunk_size])

# Preprocess each text file
for file_name in os.listdir(input_dir):
    if file_name.endswith(".txt"):
        with open(os.path.join(input_dir, file_name), "r") as f:
            text = f.read()
        text = clean_text(text)
        chunks = list(chunk_text(text))
        for i, chunk in enumerate(chunks):
            chunk_file = os.path.join(output_dir, f"{file_name.replace('.txt', '')}_chunk{i}.txt")
            with open(chunk_file, "w") as cf:
                cf.write(chunk)
        print(f"Processed: {file_name}")

print("Text preprocessing complete!")
