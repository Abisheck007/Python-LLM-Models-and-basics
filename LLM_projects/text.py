# app.py
import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load model and tokenizer
model_id = "microsoft/DialoGPT-medium"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)

st.title("DialoGPT Chatbot 🤖")

# Initialize chat history
if "history" not in st.session_state:
    st.session_state.history = []

# User input
user_input = st.text_input("You:", "")

if user_input:
    # Encode the user input
    new_input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors="pt")

    # If there is previous conversation, concatenate it
    if st.session_state.history:
        # Append new input to previous chat
        chat_history_ids = torch.cat(st.session_state.history + [new_input_ids], dim=-1)
    else:
        chat_history_ids = new_input_ids

    # Generate response
    bot_output_ids = model.generate(chat_history_ids, max_new_tokens=100, pad_token_id=tokenizer.eos_token_id)
    bot_response = tokenizer.decode(bot_output_ids[:, chat_history_ids.shape[-1]:][0], skip_special_tokens=True)

    # Append new input and bot output to history
    st.session_state.history.append(new_input_ids)
    st.session_state.history.append(bot_output_ids[:, chat_history_ids.shape[-1]:])

    # Display conversation
    st.write(f"You: {user_input}")
    st.write(f"Bot: {bot_response}")

# Display full chat (optional)
st.write("### Full Chat")
for i in range(0, len(st.session_state.history), 2):
    user_text = tokenizer.decode(st.session_state.history[i][0], skip_special_tokens=True)
    bot_text = tokenizer.decode(st.session_state.history[i+1][0], skip_special_tokens=True)
    st.write(f"You: {user_text}")
    st.write(f"Bot: {bot_text}")
