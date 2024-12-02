import pdfplumber
import os

# Define paths
current_dir = os.path.dirname(os.path.abspath(__file__))
input_dir = os.path.join(current_dir, "../raw_data")
output_dir = os.path.join(current_dir, "../processed_data")

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

# Extract text from PDFs
for file_name in os.listdir(input_dir):
    if file_name.endswith(".pdf"):
        input_file = os.path.join(input_dir, file_name)
        with pdfplumber.open(input_file) as pdf:
            text = ""
            for page in pdf.pages:
                text += page.extract_text() or ""
        output_file = os.path.join(output_dir, file_name.replace(".pdf", ".txt"))
        with open(output_file, "w") as f:
            f.write(text)
        print(f"Extracted: {output_file}")

print("PDF text extraction complete!")
