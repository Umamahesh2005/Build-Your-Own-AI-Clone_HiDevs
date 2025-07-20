from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings  # ✅ Use only this

import os
from dotenv import load_dotenv
load_dotenv()

# Load your document
loader = TextLoader("docs/knowledge.txt")
docs = loader.load()

# Split into chunks
splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(docs)

# ✅ Use HuggingFace Embeddings only
embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Save to Chroma vector store
vectordb = Chroma.from_documents(documents=chunks, embedding=embedding, persist_directory="vectorstore")

print("✅ Vector store created and saved.")
