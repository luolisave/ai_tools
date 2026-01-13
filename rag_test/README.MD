# RAG demo with CPU only

## Install
~~~
py -3.11 -m pip install langchain faiss-cpu sentence-transformers transformers
py -3.11 -m pip uninstall -y langchain langchain-community langchain-core langchain-huggingface
py -3.11 -m pip install -U langchain langchain-community langchain-core langchain-huggingface
pip install --upgrade transformers accelerate
pip install soxr
~~~

## Run DEMO 1
~~~
py -3.11 .\rag.py
~~~

## Run DEMO 2
This DEMO generate vectorstore and load vectorstore in seperate py files
~~~
py -3.11 .\c1_generate_vectorstore_file.py
py -3.11 .\c2_use_vectorstore_file.py
~~~