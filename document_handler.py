from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Your knowledge base (you can load from files later)
documents = [
    "Einstein developed the theory of relativity in the early 20th century.",
    "Python is a versatile programming language used for web development, AI, and data science.",
    "The capital city of France is Paris, known for its culture and landmarks.",
    "Streamlit is a Python library for creating interactive web apps for data science."
]

def retrieve_relevant_chunks(query, docs=documents):
    # Vectorize the documents + query
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(docs + [query])

    # Compute cosine similarity
    cosine_sim = cosine_similarity(vectors[-1], vectors[:-1])

    # Find the index of the most similar document
    top_idx = cosine_sim[0].argmax()
    return docs[top_idx]
