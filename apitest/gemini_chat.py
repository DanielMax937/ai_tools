import google.generativeai as genai
import os
import dotenv

dotenv.load_dotenv()

# Configure the SDK with your API key
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

# Create a Gemini Pro model
model = genai.GenerativeModel('gemini-2.5-pro')

# Start a chat session with an empty history
chat = model.start_chat(history=[])

print("🤖 Hello! I'm a chat bot powered by the Gemini API. Type 'exit' to end the conversation.")

# Main chat loop
while True:
    # Get user input from the command line
    user_input = input("You: ")
    
    # Check if the user wants to exit
    if user_input.lower() == 'exit':
        print("🤖 Goodbye!")
        break
        
    # Send the user's message to the model
    response = chat.send_message(user_input)
    
    # Print the model's response
    print(f"Gemini: {response.text}")