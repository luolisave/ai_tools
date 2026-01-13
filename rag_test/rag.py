from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from transformers import pipeline
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
retriever = vectorstore.as_retriever()

# 3. LLM (instruction-tuned)
pipe = pipeline(
    "text2text-generation",
    model="google/flan-t5-base",
    max_new_tokens=50,
    device_map="cuda" if torch.cuda.is_available() else "cpu"
)
llm = HuggingFacePipeline(pipeline=pipe)

# 4. Prompt
prompt = PromptTemplate.from_template(
    """Answer the question using only the context below.

Context:
{context}

Question:
{question}
"""
)

# 5. RAG chain (LCEL)
rag_chain = (
    {
        "context": retriever,
        "question": RunnablePassthrough(),
    }
    | prompt
    | llm
)

# 6. Run
result = rag_chain.invoke("How many days of annual leave are there?")
print(result)
