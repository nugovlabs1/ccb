import openai
import os
import asyncio

# Set API key
openai.api_key = "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"

# Function to process all chunks and ask GPT-4
async def ask_gpt4():
    # Define the outputs folder
    current_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(current_dir, "../outputs")

    # Concatenate all chunks into a single context string
    context = ""
    for file_name in sorted(os.listdir(output_dir)):
        if file_name.endswith(".txt"):
            with open(os.path.join(output_dir, file_name), "r") as f:
                context += f.read() + "\n\n"

    # Ensure the context fits within the model's token limit
    context = context[:7000]  # Adjust this to leave room for the question and response

    # Define the question
    question = "I live at 10511 Madera Dr Zip: 95014 and I want to rebuild my house. Where do I start?"

    # Ask GPT-4
    response = await openai.ChatCompletion.acreate(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a helpful assistant for city-related questions."},
            {"role": "user", "content": f"Context: {context}\n\nQuestion: {question}"}
        ]
    )

    # Print the answer
    print(response.choices[0].message["content"])

# Run the async function
asyncio.run(ask_gpt4())
