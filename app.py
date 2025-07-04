import os
from dotenv import load_dotenv
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from langchain_community.vectorstores import FAISS
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain.chains import RetrievalQA
from langchain.schema import HumanMessage

load_dotenv()  # load .env file

app = Flask(__name__, static_folder="static", template_folder="templates")
CORS(app)

VECTORSTORE_DIR = "faiss_index"

# Clean conflicting env vars if any
for var in ["OPENAI_API_BASE", "OPENAI_BASE_URL", "OPENAI_API_BASE_URL"]:
    os.environ.pop(var, None)

embedding = AzureOpenAIEmbeddings(
    model=os.getenv("DEPLOYMENT_NAME_EMBEDDING"),
    deployment=os.getenv("DEPLOYMENT_NAME_EMBEDDING"),
    openai_api_key=os.getenv("OPENAI_API_KEY_EMBEDDING"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT_EMBEDDING"),
    openai_api_version=os.getenv("OPENAI_API_VERSION_EMBEDDING"),
    chunk_size=1000,
    validate_base_url=False,
)

def get_vectorstore():
    if os.path.exists(VECTORSTORE_DIR):
        return FAISS.load_local(VECTORSTORE_DIR, embedding, allow_dangerous_deserialization=True)
    return None

def get_qa_chain():
    db = get_vectorstore()
    if not db:
        raise ValueError("FAISS vectorstore index not found. Please run vector.py first to create index.")
    llm = AzureChatOpenAI(
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT_CHAT"),
        openai_api_version=os.getenv("OPENAI_API_VERSION_CHAT"),
        deployment_name=os.getenv("DEPLOYMENT_NAME_CHAT"),
        openai_api_key=os.getenv("OPENAI_API_KEY_CHAT"),
    )
    retriever = db.as_retriever(search_kwargs={"k": 3})
    return RetrievalQA.from_chain_type(llm=llm, retriever=retriever, chain_type="stuff")

def get_chat_llm():
    return AzureChatOpenAI(
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT_CHAT"),
        openai_api_version=os.getenv("OPENAI_API_VERSION_CHAT"),
        deployment_name=os.getenv("DEPLOYMENT_NAME_CHAT"),
        openai_api_key=os.getenv("OPENAI_API_KEY_CHAT"),
        temperature=0.7,
        max_tokens=1000,
    )

qa_chain = get_qa_chain()
chat_llm = get_chat_llm()

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    question = data.get("question", "").strip()
    if not question:
        return jsonify({"error": "Empty question"}), 400
    try:
        answer = qa_chain.run(question)

        # Simple fallback heuristic:
        if not answer or "I don't know" in answer.lower() or len(answer.strip()) < 20:
            chat_response = chat_llm([HumanMessage(content=question)])
            answer = chat_response.content

        return jsonify({"answer": answer})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    import threading
    import webbrowser

    def open_browser():
        webbrowser.open_new("http://127.0.0.1:5000/")

    if os.environ.get("WERKZEUG_RUN_MAIN") == "true":
        threading.Timer(1.0, open_browser).start()
    app.run(debug=True, use_reloader=False)
