from langchain_ollama import OllamaLLM
model= OllamaLLM(
    model="tinyllama",
    base_url="http://127.0.0.1:11434",
    timeout=120
)
question="what is ai?"

response = model.invoke(question)
print(response)
print("Starting")

import langchain_community
print("langchain_community OK")

from langchain_community.vectorstores import FAISS
print("FAISS OK")

from langchain_community.embeddings import HuggingFaceEmbeddings, huggingface

print("Embeddings OK")
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

loader = PyPDFLoader("Thermodynamics_Questions_Answers.pdf")
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=50
)

chunks = text_splitter.split_documents(documents)

print(len(chunks))
print(chunks[0].page_content)
embeddings=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = FAISS.from_documents(
    documents=chunks,
    embedding=embeddings
)
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_ollama import OllamaLLM

retriever = db.as_retriever()

prompt = ChatPromptTemplate.from_template(
    """Use the following context to answer the question.

Context:
{context}

Question:
{question}

Answer:"""
)


rag_chain = (
    {
        "context": lambda x: retriever.invoke(x["question"]),
        "question": RunnablePassthrough(),
    }
    | prompt
    | model
)

response = rag_chain.invoke(
    {"question": "open loop vs closed loop"}
)

print(response)
