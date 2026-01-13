# RAG demo with CPU only

## Install
~~~
py -3.11 -m pip install langchain faiss-cpu sentence-transformers transformers
py -3.11 -m pip uninstall -y langchain langchain-community langchain-core langchain-huggingface
py -3.11 -m pip install -U langchain langchain-community langchain-core langchain-huggingface
pip install --upgrade transformers accelerate
pip install soxr
~~~

## Run
~~~
py -3.11 .\rag.py
~~~