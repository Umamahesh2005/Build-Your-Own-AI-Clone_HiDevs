import streamlit as st
from transformers import pipeline, set_seed

@st.cache_resource
def load_model():
    return pipeline("text-generation", model="distilgpt2")

chatbot = load_model()

def generate_prompt(question):
    return f"Q: {question}\nA:"

st.title("🤖 GPT-2 Chatbot")
st.markdown("Ask your question, and the chatbot will respond with a generated answer.")

question = st.text_input("💬 Your Question:")

if question:
    with st.spinner("Generating answer..."):
        prompt = generate_prompt(question)
        set_seed(42)  # for reproducible outputs
        result = chatbot(prompt, max_new_tokens=100, temperature=0.7, top_k=50)
        
        # Get only the generated answer part
        generated_text = result[0]["generated_text"]
        answer = generated_text.split("A:")[-1].strip()

        st.markdown("### ✅ Answer:")
        st.success(answer)






