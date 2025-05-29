import os
import shutil
from fastapi import FastAPI, File, UploadFile, HTTPException, Depends
# --- NUOVO IMPORT ---
from fastapi.middleware.cors import CORSMiddleware
# --- FINE NUOVO IMPORT ---
from pydantic import BaseModel, Field, SecretStr
from typing import List, Optional, cast
import uvicorn
from dotenv import load_dotenv

# ... (tutti gli altri tuoi import di Langchain, etc. rimangono qui) ...
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_postgres.vectorstores import PGVector
from langchain_core.documents import Document as LangchainDocument
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from unstructured.partition.auto import partition as unstructured_partition


print("DEBUG: Inizio importazioni e configurazioni...")
# ... (tutta la tua configurazione delle variabili d'ambiente rimane qui) ...
load_dotenv(override=True)
print("DEBUG: .env caricato (con override=True).")

DB_HOST_RAW = os.getenv("DB_HOST")
DB_PORT_RAW = os.getenv("DB_PORT")
DB_USER_RAW = os.getenv("DB_USER")
DB_PASSWORD_RAW = os.getenv("DB_PASSWORD")
DB_NAME_RAW = os.getenv("DB_NAME")
EMBEDDING_MODEL_NAME_RAW = os.getenv("EMBEDDING_MODEL_NAME")
COLLECTION_NAME_RAW = os.getenv("COLLECTION_NAME")
OPENROUTER_API_KEY_RAW = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_MODEL_NAME_RAW = os.getenv("OPENROUTER_MODEL_NAME")
OPENROUTER_API_BASE_RAW = os.getenv("OPENROUTER_API_BASE")

print(f"DEBUG_RAW: DB_HOST='{DB_HOST_RAW}' (tipo: {type(DB_HOST_RAW)})")
print(f"DEBUG_RAW: DB_PORT='{DB_PORT_RAW}' (tipo: {type(DB_PORT_RAW)})")
print(f"DEBUG_RAW: DB_USER='{DB_USER_RAW}' (tipo: {type(DB_USER_RAW)})")
print(f"DEBUG_RAW: DB_NAME='{DB_NAME_RAW}' (tipo: {type(DB_NAME_RAW)})")
print(f"DEBUG_RAW: EMBEDDING_MODEL_NAME='{EMBEDDING_MODEL_NAME_RAW}' (tipo: {type(EMBEDDING_MODEL_NAME_RAW)})")
print(f"DEBUG_RAW: COLLECTION_NAME='{COLLECTION_NAME_RAW}' (tipo: {type(COLLECTION_NAME_RAW)})")
print(f"DEBUG_RAW: OPENROUTER_MODEL_NAME='{OPENROUTER_MODEL_NAME_RAW}' (tipo: {type(OPENROUTER_MODEL_NAME_RAW)})")
print(f"DEBUG_RAW: OPENROUTER_API_BASE='{OPENROUTER_API_BASE_RAW}' (tipo: {type(OPENROUTER_API_BASE_RAW)})")
print("--- FINE DEBUG VALORI GREZZI ---")

def get_env_var(raw_value: Optional[str], default_value: str) -> str:
    if raw_value is not None:
        stripped_value = raw_value.strip()
        if stripped_value:
            return stripped_value
    return default_value

DB_HOST = get_env_var(DB_HOST_RAW, "fallback_host")
DB_PORT = get_env_var(DB_PORT_RAW, "5432")
DB_USER = get_env_var(DB_USER_RAW, "fallback_user")
DB_PASSWORD = get_env_var(DB_PASSWORD_RAW, "fallback_pass")
DB_NAME = get_env_var(DB_NAME_RAW, "fallback_db")
EMBEDDING_MODEL_NAME = get_env_var(EMBEDDING_MODEL_NAME_RAW, "all-MiniLM-L6-v2")
COLLECTION_NAME = get_env_var(COLLECTION_NAME_RAW, "my_knowledge_docs")
OPENROUTER_API_KEY_STR = get_env_var(OPENROUTER_API_KEY_RAW, "fallback_apikey")
OPENROUTER_MODEL_NAME = get_env_var(OPENROUTER_MODEL_NAME_RAW, "qwen/qwen3-235b-a22b:free")
OPENROUTER_API_BASE = get_env_var(OPENROUTER_API_BASE_RAW, "https://openrouter.ai/api/v1")

critical_vars_check = {
    "DB_HOST": DB_HOST, "DB_PORT": DB_PORT, "DB_USER": DB_USER,
    "DB_PASSWORD": DB_PASSWORD, "DB_NAME": DB_NAME,
    "OPENROUTER_API_KEY": OPENROUTER_API_KEY_STR
}
missing_critical_vars = [
    var_name for var_name, var_value in critical_vars_check.items()
    if var_value in ["fallback_host", "fallback_user", "fallback_pass", "fallback_db", "fallback_apikey"] or not var_value
]

if missing_critical_vars:
    error_message = f"ERRORE CRITICO: Le seguenti variabili d'ambiente essenziali non sono impostate correttamente: {', '.join(missing_critical_vars)}. Controlla il tuo file .env."
    print(error_message)
    raise ValueError(error_message)

CONNECTION_STRING = f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
print(f"DEBUG: Connection string COSTRUITA: '{CONNECTION_STRING}'")


# --- Modelli Pydantic ---
class UploadResponse(BaseModel):
    filename: str
    message: str
    chunks_added: int

class QueryRequest(BaseModel):
    question: str
    top_k: int = Field(default=3, ge=1, le=10)

class QueryResponse(BaseModel):
    question: str
    answer: str
    source_chunks: List[str]

print("DEBUG: Modelli Pydantic definiti.")

# Crea l'istanza dell'applicazione FastAPI
app = FastAPI(title="Knowledge Base API with PGVector & OpenRouter (langchain-openai)")
print("DEBUG: Istanza FastAPI creata.")


# --- CONFIGURAZIONE CORS ---
# Definisci le origini che possono accedere al tuo backend
# Sostituisci con la porta effettiva del tuo server di sviluppo frontend
origins = [
    "http://localhost", # Se il frontend è servito dalla stessa macchina senza porta specifica (raro per dev server)
    "http://localhost:3000",  # Esempio per React dev server (Create React App)
    "http://localhost:5173",  # Esempio per Vite dev server
    "http://localhost:8080",  # Altra porta comune per dev server (es. Vue CLI)
    # Aggiungi qui l'URL da cui il tuo frontend `fastapi-server-interface` viene servito durante lo sviluppo
    # Se usi Live Server in VSCode, potrebbe essere http://127.0.0.1:5500 o simile
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # Lista delle origini permesse
    allow_credentials=True, # Permette i cookie nelle richieste cross-origin (se li usi)
    allow_methods=["*"],    # Permette tutti i metodi HTTP (GET, POST, PUT, DELETE, ecc.)
    allow_headers=["*"],    # Permette tutti gli header HTTP
)
print(f"DEBUG: CORSMiddleware configurato per permettere le origini: {origins}")
# --- FINE CONFIGURAZIONE CORS ---


# --- Inizializzazione Modello di Embedding ---
# ... (come prima) ...
embeddings_model: Optional[HuggingFaceEmbeddings] = None
try:
    print(f"DEBUG: Caricamento modello embedding '{EMBEDDING_MODEL_NAME}' su CPU...")
    embeddings_model = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={'device': 'cpu'}
    )
    print(f"DEBUG: Modello embedding '{EMBEDDING_MODEL_NAME}' caricato.")
except Exception as e:
    print(f"ERRORE CRITICO: Caricamento modello embedding fallito: {e}")


# --- Funzione get_vector_store ---
# ... (come prima) ...
def get_vector_store() -> PGVector:
    print("DEBUG: Chiamata a get_vector_store (langchain-postgres).")
    if not embeddings_model: 
        print("ERRORE: embeddings_model non è stato inizializzato correttamente.")
        raise HTTPException(status_code=503, detail="Modello di embedding non inizializzato correttamente.")

    print(f"DEBUG: Tentativo di inizializzazione PGVector per collection '{COLLECTION_NAME}' (langchain-postgres)...")
    try:
        vector_store = PGVector(
            embeddings=embeddings_model,
            collection_name=COLLECTION_NAME,
            connection=CONNECTION_STRING,
        )
        print(f"DEBUG: Istanza PGVector creata/configurata per collection '{COLLECTION_NAME}'.")
    except Exception as e:
        print(f"ERRORE CRITICO: Creazione PGVector store fallita per collection '{COLLECTION_NAME}': {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=503, detail=f"Impossibile creare/connettersi al Vector Store: {e}")
    return vector_store

# --- Endpoint /upload_pdf/ ---
# ... (come prima) ...
@app.post("/upload_pdf/", response_model=UploadResponse)
async def upload_pdf(file: UploadFile = File(...), vector_store: PGVector = Depends(get_vector_store)):
    print(f"DEBUG: Chiamata a /upload_pdf/ per il file '{file.filename}'.")
    if not file.filename:
        raise HTTPException(status_code=400, detail="Nome del file mancante nell'upload.")
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Formato file non supportato. Caricare solo PDF.")
    if not embeddings_model: 
        print("ERRORE in /upload_pdf/: Modello di embedding non disponibile.")
        raise HTTPException(status_code=500, detail="Il modello di embedding non è disponibile.")

    current_file_name: str = file.filename
    safe_filename = current_file_name.replace('/', '_').replace('\\', '_').replace('..', '')
    temp_file_path = f"temp_{safe_filename}"

    try:
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        print(f"DEBUG: File '{current_file_name}' salvato temporaneamente in '{temp_file_path}'.")

        print(f"DEBUG: Parsing del file: {temp_file_path} con unstructured.")
        elements = unstructured_partition(filename=temp_file_path, strategy="auto")
        print(f"DEBUG: Numero di elementi estratti da unstructured: {len(elements)}")

        raw_text_chunks = [el.text for el in elements if hasattr(el, 'text') and el.text and el.text.strip()]
        if not raw_text_chunks:
            raise HTTPException(status_code=400, detail="Nessun testo estraibile o valido trovato nel PDF.")
        print(f"DEBUG: Numero di chunk di testo grezzo validi: {len(raw_text_chunks)}")

        full_text = "\n".join(raw_text_chunks)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        split_texts = text_splitter.split_text(full_text)

        if not split_texts:
             raise HTTPException(status_code=400, detail="Il testo estratto non ha prodotto chunk dopo lo splitting.")
        print(f"DEBUG: Numero di chunk dopo lo splitting del testo: {len(split_texts)}")

        documents = [LangchainDocument(page_content=text, metadata={"source": current_file_name}) for text in split_texts]
        
        print(f"DEBUG: Tentativo di aggiungere {len(documents)} documenti alla collection '{COLLECTION_NAME}' usando langchain-postgres...")
        vector_store.add_documents(documents=documents) 
        print(f"DEBUG: Aggiunti {len(documents)} chunk alla collection '{COLLECTION_NAME}'.")

        return UploadResponse(
            filename=current_file_name,
            message="PDF processato e aggiunto alla knowledge base (usando langchain-postgres).",
            chunks_added=len(documents)
        )
    except HTTPException as e: 
        raise e
    except Exception as e:
        print(f"ERRORE: Errore durante il processing del PDF '{current_file_name}': {e}")
        import traceback
        traceback.print_exc() 
        raise HTTPException(status_code=500, detail=f"Errore durante il processing del PDF: {str(e)}")
    finally:
        if os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
                print(f"DEBUG: File temporaneo '{temp_file_path}' rimosso.")
            except OSError as e_remove:
                print(f"ATTENZIONE: Impossibile rimuovere il file temporaneo '{temp_file_path}': {e_remove}")
        if file: 
             await file.close()


# --- Endpoint /query/ ---
# ... (come prima, con la correzione per SecretStr) ...
@app.post("/query/", response_model=QueryResponse)
async def query_knowledge_base(request: QueryRequest, vector_store: PGVector = Depends(get_vector_store)):
    print(f"DEBUG: Chiamata a /query/ con domanda: '{request.question}', top_k={request.top_k}.")
    if not embeddings_model: 
        print("ERRORE in /query/: Modello di embedding non disponibile.")
        raise HTTPException(status_code=500, detail="Il modello di embedding non è disponibile.")

    try:
        print(f"DEBUG: Ricerca di similarità per query: '{request.question}' k={request.top_k} (langchain-postgres)...")
        retrieved_docs = vector_store.similarity_search(query=request.question, k=request.top_k)

        if not retrieved_docs:
            print(f"DEBUG: Nessun documento rilevante trovato per query: '{request.question}'.")
            return QueryResponse(
                question=request.question,
                answer="Non ho trovato informazioni rilevanti nella knowledge base per rispondere.",
                source_chunks=[]
            )
        print(f"DEBUG: Trovati {len(retrieved_docs)} documenti rilevanti.")

        context_text = "\n\n---\n\n".join([doc.page_content for doc in retrieved_docs])
        source_chunks_content = [doc.page_content for doc in retrieved_docs] 

        prompt_template = ChatPromptTemplate.from_messages([
            ("system", "Sei un assistente AI che risponde a domande basandosi ESCLUSIVAMENTE sul contesto fornito. Se l'informazione non è nel contesto, rispondi 'Non ho trovato informazioni sufficienti nel contesto per rispondere'. Non inventare risposte."),
            ("human", "Contesto:\n{context}\n\nDomanda: {question}\n\nRisposta:")
        ])
        
        print(f"DEBUG_LLM: API Key (prime 10): '{str(OPENROUTER_API_KEY_STR)[:10]}...', Base URL: '{OPENROUTER_API_BASE}', Modello: '{OPENROUTER_MODEL_NAME.strip('"')}'")
        
        processed_api_key: Optional[SecretStr] = None
        if OPENROUTER_API_KEY_STR and OPENROUTER_API_KEY_STR != "fallback_apikey":
            processed_api_key = SecretStr(OPENROUTER_API_KEY_STR)

        llm = ChatOpenAI(
            model=OPENROUTER_MODEL_NAME.strip('"'),
            api_key=processed_api_key, 
            base_url=OPENROUTER_API_BASE,
            temperature=0.1 
        )
        
        rag_chain = ({"context": (lambda _: context_text), "question": RunnablePassthrough()} | prompt_template | llm | StrOutputParser())
        
        print("DEBUG: Invocazione della catena RAG con l'LLM (OpenRouter)...")
        answer = rag_chain.invoke(request.question)
        print(f"DEBUG: Risposta dall'LLM (primi 100 caratteri): '{answer[:100]}...'")

        return QueryResponse(
            question=request.question,
            answer=answer,
            source_chunks=source_chunks_content
        )
    except Exception as e:
        print(f"ERRORE: Errore durante la query '{request.question}': {e}")
        import traceback
        traceback.print_exc() 
        raise HTTPException(status_code=500, detail=f"Errore durante l'esecuzione della query: {str(e)}")

# --- Endpoint Root ---
@app.get("/")
async def root():
    print("DEBUG: Chiamata a GET /")
    return {"message": "Benvenuto nella Knowledge Base API! (usando langchain-openai & langchain-postgres)"}

# --- Esecuzione Uvicorn ---
if __name__ == "__main__":
    print("DEBUG: Script in esecuzione come __main__.")
    print("DEBUG: Tentativo di avvio Uvicorn su host 0.0.0.0 e porta 8000...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
    print("DEBUG: Uvicorn dovrebbe essere avviato (questo messaggio non verrà visualizzato se Uvicorn blocca il thread, il che è normale).")