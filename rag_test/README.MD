# RAG demo with CPU only

## Install
~~~
py -3.11 -m pip install langchain faiss-cpu sentence-transformers transformers
py -3.11 -m pip uninstall -y langchain langchain-community langchain-core langchain-huggingface
py -3.11 -m pip install -U langchain langchain-community langchain-core langchain-huggingface
pip install --upgrade transformers accelerate
pip install soxr
~~~

## Run DEMO 1 （CPU)
~~~
py -3.11 .\rag.py
~~~

## DEMO 2
### Run DEMO 2 - 1 (CPU)
This DEMO generate vectorstore and load vectorstore in seperate py files
~~~
py -3.11 .\cpu1_generate_vectorstore_file.py
py -3.11 .\cpu2_use_vectorstore_file.py
~~~
Now you can ask a question like: **my google password?**
And It should reply something like: **gogl1234567890**

### Run DEMO 2 - 1 (CUDA)
This DEMO generate vectorstore and load vectorstore in seperate py files
~~~
py -3.11 .\cuda1_generate_vectorstore_file.py
py -3.11 .\cuda2_use_vectorstore_file.py
~~~


## Notes
### Embedding models
model_name="sentence-transformers/all-mpnet-base-v2"  
vector 384
suggested max length: 256 tokens

model_name="sentence-transformers/all-MiniLM-L6-v2"
vector 768
suggested max length: 384 ～ 512 tokens

