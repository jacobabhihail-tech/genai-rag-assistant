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


# ----------- CORE FUNCTIONS -----------

def process_document(file_path):
    loader = PyPDFLoader(file_path)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    doc_chunks = text_splitter.split_documents(documents)

    texts = [chunk.page_content for chunk in doc_chunks]
    embeddings = embed_model.encode(texts)
    embeddings = np.array(embeddings)

    dimension = embeddings.shape[1]
    faiss_index = faiss.IndexFlatL2(dimension)
    faiss_index.add(embeddings)

    return faiss_index, doc_chunks


def retrieve_chunks(query, faiss_index, doc_chunks):
    query_vector = embed_model.encode([query])
    D, I = faiss_index.search(query_vector, k=min(2, len(doc_chunks)))

    retrieved_text = ""
    for i in I[0]:
        retrieved_text += doc_chunks[i].page_content + "\n"

    return retrieved_text


def generate_answer(query, context):
    content = f"""
You are an AI assistant.

Use ONLY the context below to answer the question.
If the answer is not in the context, say "I don't know".

Context:
{context}

Question:
{query}

Provide a clear and structured answer.
"""

    response = client.chat.completions.create(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        messages=[
            {"role": "user", "content": content}
        ]
    )

    return response.choices[0].message.content


# ----------- API ENDPOINTS -----------

@app.get("/")
def home():
    return {"message": "API is working"}


@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    global index, chunks

    file_location = f"temp_{file.filename}"

    with open(file_location, "wb") as f:
        f.write(await file.read())

    index, chunks = process_document(file_location)

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

    try:
        context = retrieve_chunks(query, index, chunks)
        answer = generate_answer(query, context)

        return {"answer": answer}

    except Exception as exc:
        return {"error": str(exc)}