# Spiegazione di routers/query.py:
# APIRouter: Come per upload.py.
# Import:
# Modelli Pydantic: QueryRequest, QueryResponse, QueryResponseSourceDocument.
# Dipendenze: get_settings, get_rag_llm. get_vector_store verrà chiamato manualmente perché il collection_name dipende dalla richiesta.
# Servizi: rag_service.
# @router.post("/query/", response_model=QueryResponse):
# request_body: QueryRequest: Ora l'endpoint si aspetta un corpo JSON che corrisponda al modello QueryRequest (che include question, top_k, e l'opzionale document_category).
# settings_dep, rag_llm_dep: Dipendenze iniettate da FastAPI.
# Logica collection_to_query: Simile all'upload, determina la collection da usare basandosi su request_body.document_category o sul default.
# Ottenere vector_store_instance:
# Qui c'è una sottigliezza. get_vector_store in core/dependencies.py è definito per prendere collection_name come argomento normale e settings_dep, embeddings_dep come dipendenze FastAPI.
# Quando chiamiamo get_vector_store manualmente dall'interno di un altro endpoint (come facciamo qui), FastAPI non risolve automaticamente le sue sub-dipendenze (settings_dep, embeddings_dep per quella chiamata specifica).
# Quindi, dobbiamo ottenere esplicitamente current_embeddings (chiamando get_embeddings_model con settings_dep che abbiamo già) e poi passare settings_dep e current_embeddings a get_vector_store. Questo assicura che get_vector_store riceva tutto ciò di cui ha bisogno.
# Chiamata a rag_service.get_answer_from_rag: Passa la richiesta, il vector store specifico, l'LLM e le configurazioni.
# Costruzione QueryResponse: Prende il risultato dal servizio RAG e lo mappa al modello Pydantic QueryResponse, inclusa la trasformazione dei source_documents_content in oggetti QueryResponseSourceDocument.
# Error Handling: Gestisce HTTPException e altri errori generici.


import traceback
from typing import Optional

from fastapi import APIRouter, HTTPException, Depends, Form

from models.pydantic_models import QueryRequest, QueryResponse, QueryResponseSourceDocument
from core.config import Settings # settings non viene usato direttamente qui, ma tramite Depends
from core.dependencies import get_settings, get_rag_llm, get_vector_store, get_embeddings_model # Aggiunto get_embeddings_model

from services import rag_service

from langchain_openai import ChatOpenAI
from langchain_postgres.vectorstores import PGVector
from langchain_huggingface import HuggingFaceEmbeddings # Per type hinting


router = APIRouter()
print("DEBUG: Router 'query_router' (/routers/query.py) creato.")


@router.post("/query/", response_model=QueryResponse)
async def api_query_knowledge_base(
    request_body: QueryRequest,
    settings_dep: Settings = Depends(get_settings), # settings_dep è usato per passarlo a get_vector_store
    rag_llm_dep: ChatOpenAI = Depends(get_rag_llm),
    # --- MODIFICA QUI: Iniettiamo get_embeddings_model qui ---
    embeddings_model_dep: HuggingFaceEmbeddings = Depends(get_embeddings_model)
    # --- FINE MODIFICA ---
):
    collection_to_query: str
    if request_body.document_category and request_body.document_category.strip():
        sanitized_category = "".join(c if c.isalnum() else "_" for c in request_body.document_category.strip().lower())
        collection_to_query = f"{settings_dep.DEFAULT_COLLECTION_NAME}_{sanitized_category}"
    else:
        collection_to_query = settings_dep.DEFAULT_COLLECTION_NAME

    print(f"DEBUG API (/query/): Ricevuta domanda '{request_body.question[:50]}...' per collection '{collection_to_query}'. Top k: {request_body.top_k}")

    try:
        # --- MODIFICA QUI: Passiamo embeddings_model_dep a get_vector_store ---
        vector_store_instance = get_vector_store(
            collection_name=collection_to_query,
            settings_dep=settings_dep, # Passiamo l'istanza settings_dep
            embeddings_dep=embeddings_model_dep # Passiamo l'istanza embeddings_model_dep
        )
        # --- FINE MODIFICA ---
        
        rag_result = await rag_service.get_answer_from_rag(
            request=request_body,
            vector_store=vector_store_instance,
            llm=rag_llm_dep,
            settings=settings_dep 
        )

        source_docs_for_response = [
            QueryResponseSourceDocument(page_content=doc_data.get("page_content", ""), metadata=doc_data.get("metadata", {}))
            for doc_data in rag_result.get("source_documents_content", [])
        ]
        
        return QueryResponse(
            question=request_body.question,
            answer=rag_result.get("answer", "Errore nella generazione della risposta."),
            source_documents=source_docs_for_response,
            collection_queried=collection_to_query
        )

    except HTTPException as e:
        print(f"ERRORE API (Query - HTTPException gestita): {e.detail} (status code: {e.status_code})")
        raise e
    except Exception as e:
        print(f"ERRORE API (Query - Generico non gestito) per domanda '{request_body.question}': {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Errore interno del server durante l'esecuzione della query.")