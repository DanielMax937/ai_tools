from openai import OpenAI
import os
import dotenv

dotenv.load_dotenv()

# Keep credentials in the local environment; never commit them to source control.
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL", "http://14.103.121.178/v1"),
)

try:
    # Create a chat completion request
    # chat_completion = client.chat.completions.create(
    #     messages=[
    #         {
    #             "role": "system",
    #             "content": "You are a helpful assistant that provides concise answers."
    #         },
    #         {
    #             "role": "user",
    #             "content": "What is your name?",
    #         }
    #     ],
    #     model="gpt-4o",
    # )

    # # Print the model's response
    # print(chat_completion.choices[0].message.content)

    audio_file = open("/Users/caoxiaopeng/Desktop/ai_tools/downloads/test.mp3", "rb")
    translation = client.audio.translations.create(
        model="whisper-1",
        file=audio_file,
    )

    print(translation.text)


except Exception as e:
    print(f"An error occurred: {e}")
