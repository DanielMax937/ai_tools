import google.generativeai as genai
import os
import time
import dotenv
from PIL import Image
from io import BytesIO

dotenv.load_dotenv()

# Configure the SDK with your API key from the environment variable
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

# Provide the path to your local audio file
# audio_file_path = "./downloads/test.mp3" # Using the file you provided as an example
# print(f"Uploading file: {audio_file_path}...")

# # Upload the file and get a file object
# audio_file = genai.upload_file(path=audio_file_path)
# print(f"Completed upload: {audio_file.name}")

# Wait for the file to be processed before making a generation request
# while audio_file.state.name == "PROCESSING":
#     print('Waiting for processing...')
#     time.sleep(10)
#     # Get the latest status of the file
#     audio_file = genai.get_file(audio_file.name)

# if audio_file.state.name == "FAILED":
#   raise ValueError("File processing failed.")

print(f"File is now active and ready for analysis.")

model = genai.GenerativeModel(model_name="models/gemini-2.5-pro")

# Create your prompt
# prompt = "Please listen to this podcast episode and provide a detailed summary. Identify the main speaker, key topics, and any actionable advice given."
prompt = "generate a image of a cat"
# Send the file and prompt to the model
response = model.generate_content([prompt])

# Print the result
print("\n--- 🤖 AI Analysis ---")
print(response.text)


for part in response.candidates[0].content.parts:
  if part.text is not None:
    print(part.text)
  elif part.inline_data is not None:
    image = Image.open(BytesIO((part.inline_data.data)))
    image.save("./apitest/cat.png")
    image.show()