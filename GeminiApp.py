import os
import streamlit as st
from groq import Groq
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

st.set_page_config(
    page_title="Groq AI Chatbot",
    page_icon="⚡",
    layout="centered"
)

st.title("⚡ Groq AI Chatbot")
st.caption("Powered by Groq Hardware & Llama 3.3 70B")

# Retrieve API key (from local .env or Streamlit Cloud secrets)
api_key = os.getenv("GROQ_API_KEY") or st.secrets.get("GROQ_API_KEY")

if not api_key:
    with st.sidebar:
        st.header("Configuration")
        api_key = st.text_input("Enter Groq API Key:", type="password")
        if not api_key:
            st.warning("Please provide a Groq API Key to proceed.")
            st.stop()

# Initialize Groq client
client = Groq(api_key=api_key)

# Initialize Session Chat History
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "system", "content": "You are a helpful and clear AI assistant."}
    ]

# Display prior chat history (excluding system prompt)
for message in st.session_state.messages:
    if message["role"] != "system":
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("Type your message..."):
    # Render user prompt
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Stream assistant response from Groq
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        try:
            # Send conversation history to Groq
            stream = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=st.session_state.messages,
                stream=True,
            )
            
            for chunk in stream:
                if chunk.choices[0].delta.content:
                    full_response += chunk.choices[0].delta.content
                    message_placeholder.markdown(full_response + "▌")
                    
            message_placeholder.markdown(full_response)
            st.session_state.messages.append({"role": "assistant", "content": full_response})
            
        except Exception as e:
            st.error(f"Error generating response: {e}")
