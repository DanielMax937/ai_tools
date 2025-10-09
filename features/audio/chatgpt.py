from openai import OpenAI
import os
import dotenv

dotenv.load_dotenv()

client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY_MIRACLE"),
    base_url='http://openai-proxy.miracleplus.com/v1'
)
audio_file= open("/Users/caoxiaopeng/Desktop/ai_tools/downloads/test.mp3", "rb")

transcription = client.audio.transcriptions.create(
    model="gpt-4o-transcribe", 
    file=audio_file
)

print(transcription.text)