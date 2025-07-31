from openai import OpenAI
import os
import dotenv

dotenv.load_dotenv()

# The client automatically reads the OPENAI_API_KEY environment variable.
client = OpenAI(
    # api_key=os.getenv('OPENAI_API_KEY_MIRACLE'),
    # base_url="http://openai-proxy.miracleplus.com/v1"
    api_key=os.getenv('ONEAPI_FOR_GOOGLE'),
    base_url="http://43.167.226.186:8745/v1"
)

try:
    # Create a chat completion request
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant that provides concise answers."
            },
            {
                "role": "user",
                "content": "What is your",
            }
        ],
        model="gemini-2.5-pro",
    )

    # Print the model's response
    print(chat_completion.choices[0].message.content)

except Exception as e:
    print(f"An error occurred: {e}")