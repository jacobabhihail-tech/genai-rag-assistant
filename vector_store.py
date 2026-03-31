from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Load PDF
loader = PyPDFLoader("Profile.pdf")
documents = loader.load()

# Chunking
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
chunks = text_splitter.split_documents(documents)

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Convert all chunks into embeddings
texts = [chunk.page_content for chunk in chunks]
embeddings = model.encode(texts)

# Convert to numpy array
embeddings = np.array(embeddings)

# Create FAISS index
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)

# Add embeddings to index
index.add(embeddings)

print(f"Total vectors stored: {index.ntotal}")

# 🔍 Test similarity search
query = "What are his skills?"
query_vector = model.encode([query])

# Search top 2 similar chunks
D, I = index.search(query_vector, k=2)

print("\nTop matching chunks:\n")
for i in I[0]:
    print(chunks[i].page_content)
    print("-----")