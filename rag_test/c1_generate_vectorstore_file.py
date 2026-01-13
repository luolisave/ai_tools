from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

import torch

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"Device name: {torch.cuda.get_device_name(0)}")

# 1. Documents
docs = [
    "Company annual leave is 10 days.",
    "Leave requests must be submitted 3 days in advance.",
    "Sick leave requires a doctor's note."
]

# 2. Embeddings + vector store
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"}
)
vectorstore = FAISS.from_texts(docs, embeddings)

# Save to a directory
vectorstore.save_local("vectorstore_db")