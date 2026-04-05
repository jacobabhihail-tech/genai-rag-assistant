import os
from fastapi import FastAPI, UploadFile, File
from dotenv import load_dotenv
from openai import OpenAI

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings

import faiss
import numpy as np

print("APP STARTED SUCCESSFULLY")

load_dotenv()

app = FastAPI()

# LLM setup
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL")
)

# Embedding model
embeddings_model = OpenAIEmbeddings(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL")
)

# Global storage
index = None
chunks = None
chat_history = []


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

    embeddings = embeddings_model.embed_documents(texts)
    embeddings = np.array(embeddings)

    dimension = embeddings.shape[1]

    faiss_index = faiss.IndexFlatL2(dimension)
    faiss_index.add(embeddings)

    return faiss_index, doc_chunks


def retrieve_chunks(query, faiss_index, doc_chunks):
    query_vector = embeddings_model.embed_query(query)
    query_vector = np.array([query_vector])

    D, I = faiss_index.search(query_vector, k=min(2, len(doc_chunks)))

    retrieved_text = ""
    for i in I[0]:
        retrieved_text += doc_chunks[i].page_content + "\n"

    return retrieved_text


def generate_answer(query, context):
    global chat_history

    if not context.strip():
        return "I don't know"

    system_prompt = """
You are an AI assistant.

Use ONLY the provided context to answer.
If answer is not in context, say "I don't know".
Be clear and structured.
"""

    messages = [{"role": "system", "content": system_prompt}]

    # Add past chat memory
    for q, a in chat_history:
        messages.append({"role": "user", "content": q})
        messages.append({"role": "assistant", "content": a})

    # Add current question
    messages.append({
        "role": "user",
        "content": f"Context:\n{context}\n\nQuestion:\n{query}"
    })

    response = client.chat.completions.create(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        messages=messages
    )

    answer = response.choices[0].message.content

    # Save memory
    chat_history.append((query, answer))

    return answer


# ----------- API ENDPOINTS -----------

@app.get("/")
def home():
    return {"message": "API is working"}


@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    global index, chunks, chat_history

    if not file.filename.endswith(".pdf"):
        return {"error": "Only PDF files are allowed"}

    file_location = f"temp_{file.filename}"

    with open(file_location, "wb") as f:
        f.write(await file.read())

    index, chunks = process_document(file_location)

    os.remove(file_location)

    # Reset memory on new upload
    chat_history = []

    return {
        "message": "File processed successfully",
        "chunks": len(chunks)
    }


@app.get("/ask")
def ask(query: str):
    global index, chunks

    if index is None or chunks is None:
        return {"error": "No document uploaded yet"}

    context = retrieve_chunks(query, index, chunks)
    answer = generate_answer(query, context)

    return {"answer": answer}


@app.post("/clear")
def clear_chat():
    global chat_history
    chat_history = []
    return {"message": "Chat cleared"}