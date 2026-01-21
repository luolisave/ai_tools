from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough

import transformers
from transformers import pipeline

print("transformers.__version__", transformers.__version__)


# 2. Embeddings + vector store
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2" # model_name="sentence-transformers/all-MiniLM-L6-v2"
)
embeddings._client.max_seq_length = 256
# Load the saved vectorstore
vectorstore = FAISS.load_local("vectorstore_db_cpu", embeddings, allow_dangerous_deserialization=True)# (search_kwargs={"k": 5})  will return 5 most relevant documents, default is 4
# Now you can use it
retriever = vectorstore.as_retriever()

# 3. LLM (instruction-tuned)
pipe = pipeline(
    "text2text-generation",
    model="google/flan-t5-large", # model="google/flan-t5-base"
    temperature=0.3,
    max_new_tokens=200
)
llm = HuggingFacePipeline(pipeline=pipe)

# 4. Prompt
prompt = PromptTemplate.from_template(
    """If you don't know the answer, just say that you don't know.

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
while True:
    question = input("Your question (or 'quit' to exit): ").strip()
    if question.lower() == 'quit' or question.lower() == 'exit':
        print("Exiting...")
        break
    elif question:
        # Print the context documents
        context_docs = retriever.invoke(question)
        print("Context retrieved:")
        for doc in context_docs:
            print(f"- {doc.page_content}\n")
        
        result = rag_chain.invoke(question)
        print("LLM Final Result:")
        print(result)
    else:
        print("No question entered. Try again.")
