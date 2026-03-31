import os
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

load_dotenv()

# Load PDF
loader = PyPDFLoader("Profile.pdf")  # make sure file name is correct
documents = loader.load()

# Chunking
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
chunks = text_splitter.split_documents(documents)

# Load free embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Convert first chunk into embedding
vector = model.encode(chunks[0].page_content)

# Output
print(f"Embedding vector length: {len(vector)}")
print("\nFirst 10 values of vector:\n")
print(vector[:10])