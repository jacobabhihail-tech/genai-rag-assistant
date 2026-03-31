import os
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

from openai import OpenAI

load_dotenv()

print("KEY:", os.getenv("OPENAI_API_KEY"))
print("BASE:", os.getenv("OPENAI_BASE_URL"))

# LLM setup (OpenRouter or OpenAI-compatible)
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL")
)

# Document loading (fallback if Profile.pdf is missing)
pdf_path = "Profile.pdf"
if os.path.exists(pdf_path):
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
else:
    print(f"Warning: '{pdf_path}' not found. Using fallback sample text.")
    sample_text = "John Doe is a machine learning engineer with experience in Python, NLP, and data systems."
    class SimpleDoc:
        def __init__(self, text):
            self.page_content = text

    documents = [SimpleDoc(sample_text)]

# Chunking
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = text_splitter.split_documents(documents)

# Embeddings
embed_model = SentenceTransformer("all-MiniLM-L6-v2")
texts = [chunk.page_content for chunk in chunks]
embeddings = embed_model.encode(texts)
embeddings = np.array(embeddings)

# FAISS index
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

# 🔍 User query
query = os.getenv("USER_QUERY", "What are his skills?")
query_vector = embed_model.encode([query])

D, I = index.search(query_vector, k=min(2, len(chunks)))

# Get relevant chunks
retrieved_text = ""
for i in I[0]:
    retrieved_text += chunks[i].page_content + "\n"

# model override via env var; default to a widely supported model
selected_model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# 🧠 Send to LLM
try:
    response = client.chat.completions.create(
        model=selected_model,
        messages=[
            {
                "role": "user",
                "content": f"Answer the question based on the context below:\n\n{retrieved_text}\n\nQuestion: {query}"
            }
        ]
    )
    print("\nFinal Answer:\n")
    print(response.choices[0].message.content)
except Exception as exc:
    print("Error while calling chat completion:", exc)
    print("Try setting OPENAI_MODEL to a valid OpenRouter model (e.g., gpt-4o-mini, gpt-3.5-turbo).")