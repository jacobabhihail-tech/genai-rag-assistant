import os
from fastapi import FastAPI, UploadFile, File
from dotenv import load_dotenv
from openai import OpenAI

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

load_dotenv()

app = FastAPI()

# LLM setup
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL")
)

# Global storage
index = None
chunks = None
embed_model = SentenceTransformer("all-MiniLM-L6-v2")


@app.get("/")
def home():
    return {"message": "API is working"}


@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    global index, chunks

    file_location = f"temp_{file.filename}"

    # Save file
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

    # Remove temp file
    os.remove(file_location)

    return {
        "message": "File processed successfully",
        "chunks": len(chunks)
    }


@app.get("/ask")
def ask(query: str):
    global index, chunks

    if index is None or chunks is None:
        return {"error": "No document uploaded yet"}

    # Convert query to embedding
    query_vector = embed_model.encode([query])

    # Search FAISS
    D, I = index.search(query_vector, k=min(2, len(chunks)))

    # Retrieve relevant chunks
    retrieved_text = ""
    for i in I[0]:
        retrieved_text += chunks[i].page_content + "\n"

    # Prompt
    content = f"""
You are an AI assistant.

Use ONLY the context below to answer the question.
If the answer is not in the context, say "I don't know".

Context:
{retrieved_text}

Question:
{query}

Provide a clear and structured answer.
"""

    try:
        response = client.chat.completions.create(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            messages=[
                {
                    "role": "user",
                    "content": content
                }
            ]
        )

        return {
            "answer": response.choices[0].message.content
        }

    except Exception as exc:
        return {"error": str(exc)}