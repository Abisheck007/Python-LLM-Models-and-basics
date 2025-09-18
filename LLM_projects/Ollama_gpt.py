import streamlit as st
import requests

st.title("💬 Local Chatbot with Ollama")

# Model name (you can change to "mistral", "codellama", etc.)
MODEL = "llama2"

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# User input
if prompt := st.chat_input("Say something..."):
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    # Call Ollama API
response = requests.post(
    "http://localhost:11434/api/generate",
    json={"model": MODEL, "prompt": prompt, "stream": False}
)

bot_reply = response.json()["response"]


    # Add bot message to history
st.session_state.messages.append({"role": "assistant", "content": bot_reply})

with st.chat_message("assistant"):
        st.markdown(bot_reply)
