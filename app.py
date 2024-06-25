
from flask import Flask, request, jsonify
import os
import tiktoken
import json
import warnings
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores.chroma import Chroma
from langchain.document_loaders import PyMuPDF
from langchain.text_splitter import RecursiveCharacterTextSplitter

app = Flask(__name__)

# Initialize your components
tokenizer = tiktoken.get_encoding("cl100k_base")

def tiktoken_len(text):
    tokens = tokenizer.encode(text)
    return len(tokens)

file_path = "Washington State Tenant_Landlord.pdf"
loader = PyMuPDF(file_path)
pages = loader.load_and_split()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100,
    length_function=tiktoken_len
)
text = text_splitter.split_documents(pages)

hf = HuggingFaceEmbeddings(
    model_name="jhgan/ko-sbert-nli",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)
docsearch = Chroma.from_documents(text, hf)

openai_model = ChatOpenAI(
    model_name="gpt-3.5-turbo",
    streaming=True,
    callbacks=[StreamingStdOutCallbackHandler()],
    temperature=0
)

qa = RetrievalQA.from_chain_type(
    llm=openai_model,
    chain_type="stuff",
    retriever=docsearch.as_retriever(
        search_type="mmr",
        search_kwargs={'k': 3, 'fetch_k': 10}
    ),
    return_source_documents=True
)

def document_to_dict(document):
    return {
        "page_content": document.page_content,
        "metadata": document.metadata
    }

@app.route('/ask', methods=['POST'])
def ask_question():
    data = request.get_json()
    query = data['query']
    result = qa(query)
    result_serializable = {
        "query": result["query"],
        "result": result["result"],
        "source_documents": [document_to_dict(doc) for doc in result["source_documents"]]
    }
    return jsonify(result_serializable)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
