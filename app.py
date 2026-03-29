import os
import gradio as gr
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

# Secure API Key handling
# For local use: set GEMINI_API_KEY in your environment, e.g., using python-dotenv or terminal export
# For Hugging Face Spaces: Add 'GEMINI_API_KEY' as a Secret in your Space configuration.
# DO NOT hardcode your API key into this file for security reasons!

def respond(message, history):
    api_key = os.environ.get("GEMINI_API_KEY")
    
    if not api_key:
        return "Error: GEMINI_API_KEY environment variable is not set. Please configure it in Hugging Face Space Secrets or your local environment."
        
    try:
        # Initialize the Chat Generative AI model
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            api_key=api_key,
        )
        
        # Build the conversation history from Gradio's history structure
        # Gradio history is typically a list of lists/tuples: [[user_msg1, ai_msg1], [user_msg2, ai_msg2]]
        messages = []
        
        # Optionally, you can add a system message here for specialized context
        # messages.append(SystemMessage(content="You are a helpful AI assistant."))
        
        for user_message, ai_message in history:
            messages.append(HumanMessage(content=user_message))
            messages.append(AIMessage(content=ai_message))
            
        # Add the latest user message
        messages.append(HumanMessage(content=message))
        
        # Request response from the model
        response = llm.invoke(messages)
        return response.content
        
    except Exception as e:
        return f"An error occurred while generating the response: {str(e)}"

# Define the user-friendly Gradio Chat Interface
demo = gr.ChatInterface(
    respond,
    title="Gemini 2.5 Chatbot",
    description="A generative AI chatbot built with LangChain and Google's Gemini 2.5 Flash model.",
    examples=[
        "Hello! How are you?",
        "What are transformers in AI?",
        "Can you write a short poem about the ocean?",
        "Explain quantum computing in simple terms."
    ],
    cache_examples=False,
)

if __name__ == "__main__":
    # Launch the Gradio app
    # For local execution, this will run on http://127.0.0.1:7860
    # On Hugging Face Spaces, it will automatically serve the app.
    demo.launch()
