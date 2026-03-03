from fastapi import FastAPI
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from transformers import pipeline

app = FastAPI()

# ---------- LOAD RAG COMPONENTS ON STARTUP ----------

loader = TextLoader("sample.txt")
documents = loader.load()

embeddings = HuggingFaceEmbeddings()
vectorstore = FAISS.from_documents(documents, embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

generator = pipeline("text-generation", model="distilgpt2")


# ---------- API ROUTE ----------

@app.get("/")
def home():
    return {"message": "RAG API is running 🚀"}


@app.get("/ask")
def ask_question(query: str):
    docs = retriever.invoke(query)
    context = "\n".join([doc.page_content for doc in docs])

    prompt = f"""
    Answer the question based only on the context below.

    Context:
    {context}

    Question:
    {query}

    Answer:
    """

    response = generator(prompt, max_length=150)

    return {
        "question": query,
        "answer": response[0]["generated_text"]
    }