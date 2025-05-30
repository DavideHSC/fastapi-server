import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# --- MODIFICA QUI: Decommenta e assicurati che i nomi dei moduli corrispondano ---
from routers import upload as upload_router_module # Importa il modulo upload.py
from routers import query as query_router_module   # Importa il modulo query.py
# --- FINE MODIFICA ---
from core.config import settings 

app = FastAPI(
    title="Knowledge Base API", # Modificato titolo per brevità, puoi personalizzare
    version="1.0.0",
    description="API per caricare documenti e interrogare una knowledge base tramite RAG.",
    # Potresti voler aggiungere un root_path se servi dietro un proxy con un prefisso
    # root_path="/api/v1_app_knowledge_base" 
)
print("DEBUG: Istanza FastAPI creata in main.py.")

allowed_origins = [
    "http://localhost",
    "http://localhost:3000",
    "http://localhost:5173",
    "http://localhost:8080",
    "http://127.0.0.1:5500", # Per VSCode Live Server
    # Aggiungi qui l'URL del tuo frontend effettivo se diverso
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
print(f"DEBUG: CORSMiddleware configurato per permettere le origini: {allowed_origins}")

# --- MODIFICA QUI: Includi i router importati ---
# Assicurati che ogni modulo (upload.py, query.py) definisca un oggetto chiamato 'router'
# Esempio: in upload.py -> router = APIRouter()
app.include_router(upload_router_module.router, prefix="/api/v1/documents", tags=["Gestione Documenti"])
app.include_router(query_router_module.router, prefix="/api/v1/queries", tags=["Interrogazione Knowledge Base"])
# Ho cambiato i prefissi per essere più specifici e descrittivi
print("DEBUG: Router inclusi nell'applicazione FastAPI.")
# --- FINE MODIFICA ---

@app.get("/", tags=["Root"])
async def read_root():
    return {
        "message": f"Benvenuto nella {app.title}",
        "version": app.version,
        "docs_url": app.docs_url, # Corretto per accedere all'attributo corretto
        "redoc_url": app.redoc_url # Corretto
    }

if __name__ == "__main__":
    print("DEBUG: Avvio Uvicorn da main.py...")
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )