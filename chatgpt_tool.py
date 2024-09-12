import openai
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Set up OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

def chat_with_gpt(prompt, model="gpt-3.5-turbo"):
    try:
        response = openai.ChatCompletion.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"An error occurred: {str(e)}"

if __name__ == "__main__":
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit", "bye"]:
            print("Goodbye!")
            break
        
        response = chat_with_gpt(user_input)
        print(f"ChatGPT: {response}")