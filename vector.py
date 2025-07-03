import os
import shutil
import time
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, UnstructuredExcelLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import AzureOpenAIEmbeddings

load_dotenv()  # load .env

base_path = r"C:\Users\WEC India\OneDrive - World Energy Council India\samujjal\OneDrive - World Energy Council India\Desktop\SAMUJJAL WORK\CHATBOT -WEC\WORLD ENERGY COUNCIL INDIA CHAT_BOT_ZEUS"

pdf_folder = os.path.join(base_path, "WEC_INDIA DATABASE - storage")
embedded_folder = os.path.join(base_path, "embedded_pdfs")
excel_folder = r"C:\Users\WEC India\OneDrive - World Energy Council India\samujjal\OneDrive - World Energy Council India\Desktop\SAMUJJAL WORK\CHATBOT -WEC\wec_chatbot\data\dashboard_exports"
VECTORSTORE_DIR = os.path.join(base_path, "faiss_index")

os.makedirs(embedded_folder, exist_ok=True)

embedding = AzureOpenAIEmbeddings(
    model=os.getenv("DEPLOYMENT_NAME_EMBEDDING"),
    deployment=os.getenv("DEPLOYMENT_NAME_EMBEDDING"),
    openai_api_key=os.getenv("OPENAI_API_KEY_EMBEDDING"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT_EMBEDDING"),
    openai_api_version=os.getenv("OPENAI_API_VERSION_EMBEDDING"),
    chunk_size=1000,
    validate_base_url=False,
)

def load_unprocessed_pdfs():
    docs = []
    filenames_to_move = []

    for filename in os.listdir(pdf_folder):
        if filename.lower().endswith(".pdf"):
            path = os.path.join(pdf_folder, filename)
            try:
                loaded = PyPDFLoader(path).load()
                print(f"✅ Loaded PDF {filename} with {len(loaded)} pages.")
                docs.extend(loaded)
                filenames_to_move.append(filename)
            except Exception as e:
                print(f"⚠️ Failed to load PDF {filename}: {e}")

    return docs, filenames_to_move

def load_excels():
    docs = []
    print(f"🔍 Scanning Excel folder: {excel_folder}")
    if not os.path.exists(excel_folder):
        print("❌ Excel folder does not exist!")
        return docs

    for filename in os.listdir(excel_folder):
        if filename.endswith(".xlsx") or filename.endswith(".xls"):
            path = os.path.join(excel_folder, filename)
            try:
                loaded = UnstructuredExcelLoader(path).load()
                print(f"✅ Loaded Excel: {filename}")
                docs.extend(loaded)
            except Exception as e:
                print(f"⚠️ Failed to load Excel {filename}: {e}")
    return docs

def move_to_embedded(filenames):
    for fname in filenames:
        src = os.path.join(pdf_folder, fname)
        dst = os.path.join(embedded_folder, fname)
        try:
            shutil.move(src, dst)
            print(f"📁 Moved {fname} to embedded_pdfs/")
        except Exception as e:
            print(f"⚠️ Failed to move {fname}: {e}")

def main():
    if os.path.exists(VECTORSTORE_DIR):
        print("📦 Loading existing FAISS index...")
        db = FAISS.load_local(VECTORSTORE_DIR, embedding, allow_dangerous_deserialization=True)
    else:
        print("🆕 Creating new FAISS index...")
        db = None

    new_pdf_docs, files_to_move = load_unprocessed_pdfs()
    new_excel_docs = load_excels()

    all_docs = new_pdf_docs + new_excel_docs
    texts = [doc.page_content for doc in all_docs]

    BATCH_SIZE = 30
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i:i + BATCH_SIZE]
        print(f"📤 Embedding batch {i//BATCH_SIZE + 1} with {len(batch)} texts...")
        if db is None and i == 0:
            db = FAISS.from_texts(batch, embedding)
        else:
            db.add_texts(batch)
        db.save_local(VECTORSTORE_DIR)
        time.sleep(10)

    if files_to_move:
        move_to_embedded(files_to_move)

    print("✅ All documents embedded and moved.")

if __name__ == "__main__":
    main()
