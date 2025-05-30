import os
import shutil
import traceback # Per un logging degli errori più dettagliato
from typing import Optional

from fastapi import APIRouter, File, UploadFile, HTTPException, Depends, Form
# from pydantic import BaseModel # Non è necessario importare BaseModel qui se non si definiscono nuovi modelli

# Importa i modelli Pydantic e le dipendenze
from models.pydantic_models import UploadResponse
from core.config import Settings
from core.dependencies import get_settings, get_embeddings_model 
# get_vector_store e get_metadata_tagging_llm non sono usati direttamente qui,
# ma sono usati dal servizio document_processor che viene chiamato.

# Importa le funzioni di servizio
from services import document_processor 

# Import per type hinting nella firma dell'endpoint, sebbene la dipendenza sia già tipizzata
from langchain_huggingface import HuggingFaceEmbeddings


router = APIRouter()
print("DEBUG: Router 'upload_router' (/routers/upload.py) creato.")


@router.post("/upload_pdf/", response_model=UploadResponse)
async def api_upload_pdf(
    file: UploadFile = File(..., description="Il file PDF da caricare e processare."),
    document_category: Optional[str] = Form(
        None, 
        description="Opzionale. Categoria o nome base per la collection in cui indicizzare il documento (es. 'listini', 'manuali'). Se omesso, usa la collection di default definita nelle configurazioni del server."
    ),
    settings_dep: Settings = Depends(get_settings),
    # Inietta l'istanza del modello di embedding tramite la dipendenza
    embeddings_model_dep: HuggingFaceEmbeddings = Depends(get_embeddings_model)
    # (Futuro) Per passare l'LLM di tagging:
    # llm_for_tagging_dep: ChatOpenAI = Depends(get_metadata_tagging_llm)
):
    """
    Endpoint per caricare un file PDF, processarlo (estrarre testo, creare chunk, 
    eventualmente estrarre metadati con LLM) e aggiungere gli embedding al vector store
    nella collection specificata o di default.
    """
    collection_to_use: str
    if document_category and document_category.strip():
        # Sanificazione base del nome della categoria per usarlo come parte del nome della tabella
        # Rende lowercase e sostituisce caratteri non alfanumerici con underscore.
        sanitized_category = "".join(c if c.isalnum() else "_" for c in document_category.strip().lower())
        # Crea un nome di collection univoco combinando il default con la categoria sanificata.
        collection_to_use = f"{settings_dep.DEFAULT_COLLECTION_NAME}_{sanitized_category}"
    else:
        collection_to_use = settings_dep.DEFAULT_COLLECTION_NAME
    
    print(f"DEBUG API (/upload_pdf/): Ricevuto file '{file.filename}' per la collection '{collection_to_use}'.")

    if not file.filename: # Controllo di sicurezza, anche se FastAPI dovrebbe già validare File(...)
        raise HTTPException(status_code=400, detail="Nome del file mancante nell'upload.")
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Formato file non supportato. Caricare solo PDF.")

    # Gestione del percorso temporaneo per il file uploadato
    temp_dir = "temp_uploads" # Assicurati che questa directory esista o venga creata con permessi adeguati
    os.makedirs(temp_dir, exist_ok=True) 
    
    base, ext = os.path.splitext(file.filename)
    safe_base = "".join(c if c.isalnum() else "_" for c in base)
    # Considera di aggiungere un timestamp o UUID per maggiore unicità in scenari concorrenti
    # import uuid; temp_file_path = os.path.join(temp_dir, f"{safe_base}_{uuid.uuid4().hex}{ext}")
    temp_file_path = os.path.join(temp_dir, f"{safe_base}{ext}")

    try:
        # Salva il file uploadato temporaneamente
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        print(f"DEBUG API: File '{file.filename}' salvato temporaneamente in '{temp_file_path}'.")

        # Chiamata al servizio di processamento documenti, passando le dipendenze necessarie
        num_chunks_added = await document_processor.process_and_embed_pdf(
            pdf_path=temp_file_path,
            original_filename=file.filename,
            collection_name=collection_to_use,
            settings=settings_dep,
            embeddings_model_instance=embeddings_model_dep # Passa l'istanza del modello di embedding
            # (Futuro) Passa l'LLM per il tagging:
            # llm_for_tagging_instance=llm_for_tagging_dep 
        )

        return UploadResponse(
            filename=file.filename,
            message=f"PDF '{file.filename}' processato e aggiunto con successo alla collection '{collection_to_use}'.",
            chunks_added=num_chunks_added,
            collection_used=collection_to_use
        )

    except HTTPException as e: # Rilancia le eccezioni HTTP specifiche per una risposta corretta al client
        print(f"ERRORE API (HTTPException gestita): {e.detail} (status code: {e.status_code})")
        raise e
    except ValueError as ve: # Es. per "Nessun contenuto testuale valido estratto"
        print(f"ERRORE API (ValueError) durante il processing del PDF '{file.filename}': {ve}")
        traceback.print_exc()
        raise HTTPException(status_code=400, detail=str(ve)) # Restituisce un 400 Bad Request
    except Exception as e: # Gestisce altri errori imprevisti
        print(f"ERRORE API (Generico non gestito) durante il processing del PDF '{file.filename}': {e}")
        traceback.print_exc() # Stampa il traceback completo per debug sul server
        raise HTTPException(status_code=500, detail=f"Errore interno del server durante il processing del PDF.")
    finally:
        # Pulizia del file temporaneo
        if os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
                print(f"DEBUG API: File temporaneo '{temp_file_path}' rimosso.")
            except OSError as e_remove:
                # Logga l'errore ma non far fallire la richiesta per questo,
                # dato che l'operazione principale potrebbe essere andata a buon fine.
                print(f"ATTENZIONE API: Impossibile rimuovere il file temporaneo '{temp_file_path}': {e_remove}")
        # Chiudi il file caricato da FastAPI (importante!)
        if file: # Assicurati che 'file' sia definito
            await file.close()