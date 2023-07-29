import os
from flask import Flask, request, make_response
from flask_cors import CORS
from werkzeug.utils import secure_filename
from supabase import create_client, Client

from PyPDF2 import PdfReader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.docstore.document import Document
from dotenv import load_dotenv

import pickle

load_dotenv()

UPLOAD_FOLDER = './traindata'
ALLOWED_EXTENSIONS = {'pdf'}

app = Flask(__name__)
CORS(app)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

url: str = os.environ.get('SUPABASE_URL')
key: str = os.environ.get('SUPABASE_KEY')
supabase: Client = create_client(url, key)

chain = load_qa_chain(OpenAI(temperature=0), chain_type="stuff")
docsearch = FAISS.from_documents([Document(page_content="This is ZK-Rollup Crypto Info Data.\n\n")], OpenAIEmbeddings())

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/api/upload', methods=['POST'])
def upload():
    global reader, raw_text, texts, embeddings, docsearch
    if 'file' not in request.files:
        return {"state": "error", "message": "No file part"}
    file = request.files['file']
    uuid = request.form['uuid']
    if file.filename == '':
        return {"state": "error", "message": "No selected file"}
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        print(filename)
        reader = PdfReader(file, filename)
        raw_text = ''
        for i, page in enumerate(reader.pages):
            text = page.extract_text()
            if text:
                raw_text += text
        text_splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=200, length_function=len)
        texts = text_splitter.split_text(raw_text)

        if os.path.exists("./store/" + uuid + "/index.faiss"):
            docsearch = FAISS.load_local("./store/" + uuid, OpenAIEmbeddings())
            for text in texts:
                docsearch.add_documents([Document(page_content=text)])
        else:
            docsearch = FAISS.from_texts(texts, OpenAIEmbeddings())
        docsearch.save_local("./store/" + uuid)

        # Summarize PDF
        query = "Please summarize the PDF document in detail. Your summary should cover the main points and arguments presented in the document, as well as any supporting evidence or examples. Please provide a clear and concise summary that accurately represents the content of the document. Your summary should be in with 1, 2, 3, ... detailed numbers."
        docs = docsearch.similarity_search(query)
        completion = chain.run(input_documents=docs, question=query)
        print(completion)
        return {"state": "success", "answer": completion}
    
    return {"state": "error", "message": "Invalid file format"}

@app.route('/api/chat', methods=['POST'])
def chat():
    query = request.get_json()
    prompt = query['prompt']
    uuid = query['uuid']
    if os.path.exists("./store/" + uuid + "/index.faiss"):
        docsearch = FAISS.load_local("./store/" + uuid, OpenAIEmbeddings())
    docs = docsearch.similarity_search(prompt)
    completion = chain.run(input_documents=docs, question=prompt)
    print(completion)
    return {"answer": completion }

if __name__ == '__main__':
    app.run(debug=True)