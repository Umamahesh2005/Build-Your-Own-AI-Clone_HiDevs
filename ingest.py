from langchain_community.document_loaders import DirectoryLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter

# 1. Load docs
loader = DirectoryLoader("docs", glob="**/*.txt")
docs = loader.load()

# 2. Chunk
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(docs)

# 3. Embed
emb = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# 4. Create vectorstore
Chroma.from_documents(chunks, emb, persist_directory="vectorstore")

print("✅ vectorstore/ created.")
