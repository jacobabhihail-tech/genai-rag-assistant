import os
from fastapi import FastAPI, UploadFile, File
from dotenv import load_dotenv
from openai import OpenAI

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings

import faiss
import numpy as np

print("APP STARTING...")

load_dotenv()

app = FastAPI()

# LLM setup
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL")
)

# Embedding model (API-based, lightweight)
embeddings_model = OpenAIEmbeddings(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL")
)

# Global storage
index = None
chunks = None


# ----------- CORE FUNCTIONS -----------

def process_document(file_path):
    try:
        loader = PyPDFLoader(file_path)
        documents = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )
        doc_chunks = text_splitter.split_documents(documents)

        if len(doc_chunks) == 0:
            raise ValueError("No content found in document")

        texts = [chunk.page_content for chunk in doc_chunks]

        # API embeddings (no local model)
        embeddings = embeddings_model.embed_documents(texts)
        embeddings = np.array(embeddings)

        dimension = embeddings.shape[1]
        faiss_index = faiss.IndexFlatL2(dimension)
        faiss_index.add(embeddings)

        return faiss_index, doc_chunks

    except Exception as e:
        raise RuntimeError(f"Error processing document: {str(e)}")


def retrieve_chunks(query, faiss_index, doc_chunks):
    try:
        if not query.strip():
            raise ValueError("Query cannot be empty")

        # API embedding for query
        query_vector = embeddings_model.embed_query(query)
        query_vector = np.array([query_vector])

        D, I = faiss_index.search(query_vector, k=min(2, len(doc_chunks)))

        retrieved_text = ""
        for i in I[0]:
            retrieved_text += doc_chunks[i].page_content + "\n"

        return retrieved_text

    except Exception as e:
        raise RuntimeError(f"Error retrieving chunks: {str(e)}")


def generate_answer(query, context):
    try:
        if not context.strip():
            return "I don't know"

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

    except Exception as e:
        raise RuntimeError(f"Error generating answer: {str(e)}")


# ----------- API ENDPOINTS -----------

@app.get("/")
def home():
    return {"message": "API is working"}


@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    global index, chunks

    try:
        if not file.filename.endswith(".pdf"):
            return {"error": "Only PDF files are allowed"}

        file_location = f"temp_{file.filename}"

        with open(file_location, "wb") as f:
            f.write(await file.read())

        index, chunks = process_document(file_location)

        os.remove(file_location)

        return {
            "message": "File processed successfully",
            "chunks": len(chunks)
        }

    except Exception as e:
        return {"error": str(e)}


@app.get("/ask")
def ask(query: str):
    global index, chunks

    try:
        if index is None or chunks is None:
            return {"error": "No document uploaded yet"}

        context = retrieve_chunks(query, index, chunks)
        answer = generate_answer(query, context)

        return {"answer": answer}

    except Exception as e:
        return {"error": str(e)}