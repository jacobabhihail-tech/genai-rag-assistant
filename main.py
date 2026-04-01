from fastapi import FastAPI, UploadFile, File
import os

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

app = FastAPI()

# Global storage (temporary for now)
index = None
chunks = None
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

@app.get("/")
def home():
    return {"message": "API is working"}

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    global index, chunks

    # Save uploaded file
    file_location = f"temp_{file.filename}"
    with open(file_location, "wb") as f:
        f.write(await file.read())

    # Load PDF
    loader = PyPDFLoader(file_location)
    documents = loader.load()

    # Chunking
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = text_splitter.split_documents(documents)

    # Embeddings
    texts = [chunk.page_content for chunk in chunks]
    embeddings = embed_model.encode(texts)
    embeddings = np.array(embeddings)

    # FAISS index
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    # Cleanup
    os.remove(file_location)

    return {
        "message": "File processed successfully",
        "chunks": len(chunks)
    }