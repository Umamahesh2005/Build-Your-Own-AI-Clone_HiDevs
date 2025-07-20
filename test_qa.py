from transformers import pipeline

# Load a lightweight QA model
qa = pipeline("question-answering", model="distilbert-base-uncased-distilled-squad")

# Simulate a document chunk (normally you'd retrieve this from your vectorstore)
def retrieve_relevant_chunks(query):
    return "An AI clone is a personalized virtual assistant that mimics human interactions using machine learning and language models."

# Ask a question
question = "How can I build my own AI clone?"
result = qa(question=question, context=retrieve_relevant_chunks(question))

# Print the answer
print("Answer:", result['answer'])
