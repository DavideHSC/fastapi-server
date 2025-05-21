import os
import shutil
from fastapi import FastAPI, File, UploadFile, HTTPException, Depends
from pydantic import BaseModel, Field
from typing import List, Optional, cast # Importa cast
import uvicorn
from dotenv import load_dotenv

# Langchain imports (rimangono invariati)
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores.pgvector import PGVector
from langchain_community.docstore.document import Document as LangchainDocument
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.chat_models.openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

from unstructured.partition.auto import partition as unstructured_partition

load_dotenv()

# --- Configurazione ---
DB_HOST_ENV = os.getenv("DB_HOST")
DB_PORT_ENV = os.getenv("DB_PORT")
DB_USER_ENV = os.getenv("DB_USER")
DB_PASSWORD_ENV = os.getenv("DB_PASSWORD")
DB_NAME_ENV = os.getenv("DB_NAME")
EMBEDDING_MODEL_NAME_ENV = os.getenv("EMBEDDING_MODEL_NAME")
COLLECTION_NAME_ENV = os.getenv("COLLECTION_NAME")
OPENROUTER_API_KEY_ENV = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_MODEL_NAME_ENV = os.getenv("OPENROUTER_MODEL_NAME")
OPENROUTER_API_BASE_ENV = "https://openrouter.ai/api/v1"

# Controllo robusto delle variabili d'ambiente
env_vars_check = {
    "DB_HOST": DB_HOST_ENV, "DB_PORT": DB_PORT_ENV, "DB_USER": DB_USER_ENV,
    "DB_PASSWORD": DB_PASSWORD_ENV, "DB_NAME": DB_NAME_ENV,
    "EMBEDDING_MODEL_NAME": EMBEDDING_MODEL_NAME_ENV,
    "COLLECTION_NAME": COLLECTION_NAME_ENV,
    "OPENROUTER_API_KEY": OPENROUTER_API_KEY_ENV,
    "OPENROUTER_MODEL_NAME": OPENROUTER_MODEL_NAME_ENV
}
missing_vars = [var_name for var_name, var_value in env_vars_check.items() if not var_value]

if missing_vars:
    raise ValueError(f"Le seguenti variabili d'ambiente non sono impostate: {', '.join(missing_vars)}. Controlla il tuo file .env")

# Assegnazione dopo il controllo, usando cast per Pylance
DB_HOST: str = cast(str, DB_HOST_ENV)
DB_PORT: str = cast(str, DB_PORT_ENV)
DB_USER: str = cast(str, DB_USER_ENV)
DB_PASSWORD: str = cast(str, DB_PASSWORD_ENV)
DB_NAME: str = cast(str, DB_NAME_ENV)
EMBEDDING_MODEL_NAME: str = cast(str, EMBEDDING_MODEL_NAME_ENV)
COLLECTION_NAME: str = cast(str, COLLECTION_NAME_ENV)
OPENROUTER_API_KEY: str = cast(str, OPENROUTER_API_KEY_ENV)
OPENROUTER_MODEL_NAME: str = cast(str, OPENROUTER_MODEL_NAME_ENV)
# OPENROUTER_API_BASE è già una stringa letterale, quindi non necessita di cast
OPENROUTER_API_BASE: str = OPENROUTER_API_BASE_ENV

CONNECTION_STRING = f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

# ... (il resto del codice rimane invariato) ...
# --- Modelli Pydantic ---
class UploadResponse(BaseModel):
    filename: str
    message: str
    chunks_added: int

class QueryRequest(BaseModel):
    question: str
    top_k: int = Field(default=3)

class QueryResponse(BaseModel):
    question: str
    answer: str
    source_chunks: List[str]

app = FastAPI(title="Knowledge Base API with PGVector & OpenRouter")

try:
    embeddings_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    print(f"Modello di embedding '{EMBEDDING_MODEL_NAME}' caricato.")
except Exception as e:
    print(f"Errore durante il caricamento del modello di embedding: {e}")
    embeddings_model = None

def get_vector_store():
    if not embeddings_model:
        raise HTTPException(status_code=500, detail="Modello di embedding non inizializzato.")
    try:
        store = PGVector(
            connection_string=CONNECTION_STRING,
            embedding_function=embeddings_model,
            collection_name=COLLECTION_NAME,
        )
        return store
    except Exception as e:
        print(f"Errore nella connessione a PGVector: {e}")
        raise HTTPException(status_code=503, detail=f"Impossibile connettersi al Vector Store: {e}")

@app.post("/upload_pdf/", response_model=UploadResponse)
async def upload_pdf(file: UploadFile = File(...), vector_store: PGVector = Depends(get_vector_store)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="Nome del file mancante nell'upload.")
    
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Formato file non supportato. Caricare solo PDF.")

    if not embeddings_model:
        raise HTTPException(status_code=500, detail="Il modello di embedding non è disponibile.")

    current_file_name: str = file.filename
    temp_file_path = f"temp_{current_file_name}"
    
    try:
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        print(f"Parsing del file: {temp_file_path}")
        elements = unstructured_partition(filename=temp_file_path, strategy="auto")
        print(f"Numero di elementi estratti da unstructured: {len(elements)}")

        raw_text_chunks = [el.text for el in elements if el.text.strip()]
        if not raw_text_chunks:
            raise HTTPException(status_code=400, detail="Nessun testo estraibile trovato nel PDF.")

        full_text = "\n".join(raw_text_chunks)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        split_texts = text_splitter.split_text(full_text)
        
        if not split_texts:
             raise HTTPException(status_code=400, detail="Il testo estratto non ha prodotto chunk dopo lo splitting.")

        print(f"Numero di chunk dopo lo splitting: {len(split_texts)}")

        documents = [LangchainDocument(page_content=text, metadata={"source": current_file_name}) for text in split_texts]

        vector_store.add_documents(documents)
        print(f"Aggiunti {len(documents)} chunk al vector store '{COLLECTION_NAME}'.")

        return UploadResponse(
            filename=current_file_name,
            message="PDF processato e aggiunto alla knowledge base.",
            chunks_added=len(documents)
        )
    except HTTPException as e:
        raise e
    except Exception as e:
        print(f"Errore durante il processing del PDF: {e}")
        raise HTTPException(status_code=500, detail=f"Errore durante il processing del PDF: {str(e)}")
    finally:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
        await file.close()

@app.post("/query/", response_model=QueryResponse)
async def query_knowledge_base(request: QueryRequest, vector_store: PGVector = Depends(get_vector_store)):
    if not embeddings_model:
        raise HTTPException(status_code=500, detail="Il modello di embedding non è disponibile.")

    try:
        print(f"Ricerca di similarità per la query: '{request.question}' con k={request.top_k}")
        retrieved_docs = vector_store.similarity_search(request.question, k=request.top_k)

        if not retrieved_docs:
            return QueryResponse(
                question=request.question,
                answer="Non ho trovato informazioni rilevanti nella knowledge base per rispondere alla tua domanda.",
                source_chunks=[]
            )

        context_text = "\n\n---\n\n".join([doc.page_content for doc in retrieved_docs])
        source_chunks_content = [doc.page_content for doc in retrieved_docs]

        prompt_template = ChatPromptTemplate.from_messages([
            ("system", "Sei un assistente AI che risponde a domande basandosi ESCLUSIVAMENTE sul contesto fornito. Se l'informazione non è nel contesto, rispondi 'Non ho trovato informazioni sufficienti nel contesto per rispondere'. Non inventare risposte."),
            ("human", "Contesto:\n{context}\n\nDomanda: {question}\n\nRisposta:")
        ])

        llm = ChatOpenAI(
            model=OPENROUTER_MODEL_NAME,
            api_key=OPENROUTER_API_KEY,
            base_url=OPENROUTER_API_BASE,
            temperature=0.1,
            max_tokens=500
        )
        
        rag_chain = (
            {"context": (lambda x: context_text), "question": RunnablePassthrough()}
            | prompt_template
            | llm
            | StrOutputParser()
        )
        
        print("Invocazione della catena RAG con l'LLM...")
        answer = rag_chain.invoke(request.question)

        return QueryResponse(
            question=request.question,
            answer=answer,
            source_chunks=source_chunks_content
        )
    except Exception as e:
        print(f"Errore durante la query: {e}")
        raise HTTPException(status_code=500, detail=f"Errore durante l'esecuzione della query: {str(e)}")

@app.get("/")
async def root():
    return {"message": "Benvenuto nella Knowledge Base API!"}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)