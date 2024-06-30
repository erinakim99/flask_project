from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from werkzeug.utils import secure_filename
import os
import openai
import tiktoken
import json
import warnings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.chat_models import ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

app = Flask(__name__)
CORS(app)

# Ignore warnings
warnings.simplefilter("ignore")

# Set OpenAI API key

env_config = os.getenv("APP_SETTINGS", "config.DevelopmentConfig")
app.config.from_object(env_config)
os.environ["OPENAI_API_KEY"] = app.config.get("SECRET_KEY")

# Set up file upload configuration
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ignore warnings
warnings.simplefilter("ignore")

# Initialize tokenizer
tokenizer = tiktoken.get_encoding("cl100k_base")

def tiktoken_len(text):
    tokens = tokenizer.encode(text)
    return len(tokens)

# Store processed documents and search index
documents = []
docsearch = None

# Helper function to process documents
def document_to_dict(document):
    return {
        "page_content": document.page_content,
        "metadata": document.metadata
    }

@app.route('/upload', methods=['POST'])
def upload_file():
    global docsearch

    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']

    
    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Process the uploaded document
        loader = PyPDFLoader(filepath)
        pages = loader.load_and_split()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100,
            length_function=tiktoken_len
        )
        text = text_splitter.split_documents(pages)

        hf = HuggingFaceEmbeddings(
            model_name="sentence-transformers/nli-roberta-base-v2",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True}
        )
        docsearch = Chroma.from_documents(text, hf)

        processed_docs = [document_to_dict(doc) for doc in text]
        documents.extend(processed_docs)  # Store processed documents

        return jsonify({"message": "File uploaded and processed successfully"}), 200

@app.route('/ask', methods=['POST'])
def ask_question():
    global docsearch

    if docsearch is None:
        return jsonify({"error": "No documents available for searching"}), 400

    data = request.get_json()
    query = data['query']

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

    result = qa(query)
    result_serializable = {
        "query": result["query"],
        "result": result["result"],
        "source_documents": [document_to_dict(doc) for doc in result["source_documents"]]
    }
    return jsonify(result_serializable)

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

if __name__ == "__main__":
    import bjoern
    bjoern.run(app, "127.0.0.1", 5000)
