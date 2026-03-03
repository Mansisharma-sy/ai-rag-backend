from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from transformers import pipeline

# 1️⃣ Load document
loader = TextLoader("sample.txt")
documents = loader.load()

# 2️⃣ Create embeddings
embeddings = HuggingFaceEmbeddings()

# 3️⃣ Create vector store
vectorstore = FAISS.from_documents(documents, embeddings)

# 4️⃣ Create retriever
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

# 5️⃣ Ask question
query = "What is Artificial Intelligence?"
docs = retriever.invoke(query)c

context = "\n".join([doc.page_content for doc in docs])

# 6️⃣ Load small model
generator = pipeline("text-generation", model="distilgpt2")

prompt = f"""
Answer the question based only on the context below.

Context:
{context}

Question:
{query}

Answer:
"""

response = generator(prompt, max_length=150)

print("\nAnswer:\n")
print(response[0]["generated_text"])