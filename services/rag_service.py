# Spiegazione di services/rag_service.py (Scheletro):
# Import: Importa le classi LangChain necessarie, i modelli Pydantic e le configurazioni.
# get_answer_from_rag function:
# Argomenti:
# request: QueryRequest: L'oggetto della richiesta contenente la domanda e top_k.
# vector_store: PGVector: L'istanza di PGVector già configurata per la collection corretta (verrà passata dal router).
# llm: ChatOpenAI: L'istanza dell'LLM già configurata (verrà passata dal router).
# settings: Settings: Le configurazioni globali.
# Logica:
# Ricerca di Similarità: Esegue vector_store.similarity_search().
# Gestione Nessun Documento: Se non vengono trovati documenti, restituisce una risposta predefinita.
# Preparazione Contesto: Concatena il page_content dei documenti recuperati. Prepara anche source_documents_content per restituire i dettagli dei chunk (inclusi i metadati) al client.
# Prompt Template: Definisce il prompt per l'LLM. È simile a quello che avevi, ma incapsulato qui.
# RAG Chain: Costruisce una catena LangChain (LCEL).
# Usa un dizionario per passare context e question al prompt.
# RunnablePassthrough() per question indica che la domanda originale dell'utente verrà passata direttamente.
# Il risultato passa attraverso il prompt, poi l'LLM, e infine StrOutputParser() per ottenere una risposta testuale.
# Invocazione: Invoca la catena con la domanda.
# Restituisce: Un dizionario con answer e source_documents_content. Il router chiamante userà questo dizionario per costruire l'oggetto QueryResponse.
# Error Handling: Un blocco try...except generico per catturare errori durante il processo.

import traceback
from typing import List, Dict, Any, Optional

from langchain_core.documents import Document as LangchainDocument
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI # Per l'LLM delle query
from langchain_postgres.vectorstores import PGVector # Per interagire con il vector store

from models.pydantic_models import QueryRequest # Per il tipo di richiesta
from core.config import Settings # Per accedere alle configurazioni

# (Futuro) Per strategie di retrieval più avanzate:
# from langchain.retrievers import ...

print("DEBUG: Modulo 'rag_service' caricato.")


async def get_answer_from_rag(
    request: QueryRequest, # Oggetto QueryRequest con domanda e top_k
    vector_store: PGVector, # Istanza del vector store per la collection corretta
    llm: ChatOpenAI,        # Istanza dell'LLM configurato
    settings: Settings      # Configurazioni globali
) -> Dict[str, Any]: # Restituisce un dizionario (che diventerà QueryResponse)
    """
    Esegue una ricerca di similarità nel vector store, costruisce un prompt
    con il contesto recuperato e interroga l'LLM per ottenere una risposta.

    Args:
        request: L'oggetto QueryRequest contenente la domanda e top_k.
        vector_store: L'istanza PGVector configurata per la collection da interrogare.
        llm: L'istanza ChatOpenAI configurata per le query.
        settings: Le configurazioni dell'applicazione.

    Returns:
        Un dizionario contenente 'answer' e 'source_documents_content'.
    
    Raises:
        Exception: Se si verifica un errore durante il processo RAG.
    """
    print(f"SERVICE (rag_service): Inizio processo RAG per domanda '{request.question[:50]}...' sulla collection '{vector_store.collection_name}'.")

    try:
        # 1. Ricerca di Similarità
        print(f"SERVICE: Esecuzione similarity_search con k={request.top_k}...")
        retrieved_docs: List[LangchainDocument] = vector_store.similarity_search(
            query=request.question, 
            k=request.top_k
        )
        print(f"SERVICE: Recuperati {len(retrieved_docs)} documenti.")

        if not retrieved_docs:
            print("SERVICE: Nessun documento rilevante trovato.")
            return {
                "answer": "Non ho trovato informazioni rilevanti nella knowledge base per rispondere alla tua domanda.",
                "source_documents_content": []
            }
        
        # 2. Preparazione del Contesto per l'LLM
        context_text = "\n\n---\n\n".join([doc.page_content for doc in retrieved_docs])
        # Modificato per passare la lista di dizionari come definito in QueryResponseSourceDocument (più o meno)
        source_documents_for_response = [{"page_content": doc.page_content, "metadata": doc.metadata if doc.metadata else {}} for doc in retrieved_docs]
        
        # 3. Definizione del Prompt Template
        prompt_template_str = """Sei un assistente AI che risponde a domande basandosi ESCLUSIVAMENTE sul contesto fornito.
Non usare conoscenze esterne. Se l'informazione non è nel contesto, rispondi 'Non ho trovato informazioni sufficienti nel contesto per rispondere'.
Non inventare o inferire risposte oltre a quanto strettamente supportato dal contesto.

Contesto:
{context}

Domanda: {question}

Risposta:"""
        
        prompt = ChatPromptTemplate.from_template(prompt_template_str)
        
        # 4. Costruzione e Invocazione della RAG Chain
        rag_chain = (
            {"context": (lambda _: context_text), "question": RunnablePassthrough()} 
            | prompt
            | llm
            | StrOutputParser()
        )
        
        print("SERVICE: Invocazione della catena RAG con l'LLM...")
        answer = rag_chain.invoke(request.question)
        print(f"SERVICE: Risposta dall'LLM (primi 100 caratteri): '{answer[:100]}...'")

        return {
            "answer": answer,
            "source_documents_content": source_documents_for_response
        }

    except Exception as e:
        print(f"SERVICE ERRORE: Errore durante il processo RAG per la domanda '{request.question}': {e}")
        traceback.print_exc() # Ora traceback è definito
        raise Exception(f"Errore nel servizio RAG: {e}") from e