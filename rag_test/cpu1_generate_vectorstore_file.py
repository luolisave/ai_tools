from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from transformers import pipeline

# 1. Documents
docs = [
    "Company annual leave is 10 days.",
    "Leave requests must be submitted 3 days in advance.",
    "Sick leave requires a doctor's note.",
    "google account is goglepass1234567890",
    "facebook account is facepassdsfdsafsa",
    "bank account is bank778bankpass8414"
]

# 2. Embeddings + vector store
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2" # model_name="sentence-transformers/all-MiniLM-L6-v2"
)
embeddings._client.max_seq_length = 256

vectorstore = FAISS.from_texts(docs, embeddings)

# Save to a directory
vectorstore.save_local("vectorstore_db_cpu")