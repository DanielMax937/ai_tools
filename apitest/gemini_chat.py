from google import genai
from google.genai import types
import os
import dotenv

dotenv.load_dotenv()
http_options = types.HttpOptions(
    base_url="http://43.167.226.186:8745"
)
# Configure the SDK with your API key
client = genai.Client(api_key=os.environ["ONEAPI_FOR_GOOGLE"], http_options=http_options)
# Create a Gemini Pro model
response = client.models.generate_content(
    model='gemini-2.5-pro', contents='Why is the sky blue?'
)
print(response.text)