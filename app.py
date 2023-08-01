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
from langchain.document_loaders import PyPDFLoader
from langchain.chains import RetrievalQA
from dotenv import load_dotenv

load_dotenv()

UPLOAD_FOLDER = './traindata'
ALLOWED_EXTENSIONS = {'pdf'}

app = Flask(__name__)
CORS(app)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

url: str = os.environ.get('SUPABASE_URL')
key: str = os.environ.get('SUPABASE_KEY')
supabase: Client = create_client(url, key)

chain = load_qa_chain(llm=OpenAI(temperature=0), chain_type="stuff")
db = FAISS.from_documents([Document(page_content="This is ZK-Rollup Crypto Info Data.\n\n")], OpenAIEmbeddings())

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/api/upload', methods=['POST'])
def upload():
    global texts, db
    if 'file' not in request.files:
        return {"state": "error", "message": "No file part"}
    file = request.files['file']
    uuid = request.form['uuid']
    if file.filename == '':
        return {"state": "error", "message": "No selected file"}
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        print(filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], 'tmp.pdf'))
        loader = PyPDFLoader(os.path.join(app.config['UPLOAD_FOLDER'], 'tmp.pdf'))
        documents = loader.load()
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        texts = text_splitter.split_documents(documents)

        if os.path.exists("./store/" + uuid + "/index.faiss"):
            db = FAISS.load_local("./store/" + uuid, OpenAIEmbeddings())
        else:
            db = FAISS.from_documents(texts, OpenAIEmbeddings())
        db.save_local("./store/" + uuid)

        # Summarize PDF
        query = "Please summarize the PDF document in detail. Your summary should cover the main points and arguments presented in the document, as well as any supporting evidence or examples. Please provide a clear and concise summary that accurately represents the content of the document. Please split summaries with segments. each segment should be numbered in with 1, 2, 3, ... ."
        retriever = db.as_retriever(search_type="similarity", search_kwargs={"k":2})
        qa = RetrievalQA.from_chain_type(llm=OpenAI(), chain_type="stuff", retriever=retriever, return_source_documents=False)
        completion = qa({'query': query})
        return {"state": "success", "answer": completion.get('result')}
    
    return {"state": "error", "message": "Invalid file format"}

@app.route('/api/chat', methods=['POST'])
def chat():
    query = request.get_json()
    prompt = query['prompt']
    uuid = query['uuid']
    if os.path.exists("./store/" + uuid + "/index.faiss"):
        db = FAISS.load_local("./store/" + uuid, OpenAIEmbeddings())
    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k":2})
    qa = RetrievalQA.from_chain_type(llm=OpenAI(), chain_type="stuff", retriever=retriever, return_source_documents=True)
    completion = qa({'query': prompt})
    metadata = []
    for source_document in completion['source_documents']:
        metadata.append({'page': source_document.metadata['page'], 'ref': source_document.page_content[:200]})
    return {"state": "success", "answer": completion.get('result'), "metadata": metadata }

if __name__ == '__main__':
    app.run(debug=True)