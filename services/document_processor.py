# Spiegazione di services/document_processor.py:
# Import: Importa le librerie necessarie, i modelli Pydantic, e le funzioni di dipendenza.
# process_and_embed_pdf function:
# Argomenti: Prende pdf_path, original_filename, collection_name e l'oggetto settings.
# Ottenere Dipendenze: All'inizio, chiama get_embeddings_model() (e in futuro get_metadata_tagging_llm()) usando l'oggetto settings passato. Questo assicura che usiamo le istanze cachate/configurate correttamente dei modelli.
# Parsing PDF con unstructured: Estrae gli elementi grezzi dal PDF. Ho aggiunto languages=['ita', 'eng'] come suggerimento a unstructured per migliorare il parsing di documenti italiani/inglesi.
# (FASE FUTURA) Estrazione Metadati LLM: Ho inserito un blocco commentato dove andrà la logica per usare OpenAIMetadataTagger. Per ora, creiamo base_doc_metadata con valori di default/None per i campi che l'LLM dovrebbe popolare. Ho usato DocumentChunkMetadata per validare e strutturare questi metadati di base.
# Chunking:
# Itera sui "documenti" da processare (per ora, solo uno: l'intero testo del PDF).
# Usa RecursiveCharacterTextSplitter. Ho reso chunk_size e chunk_overlap configurabili tramite l'oggetto settings (dovrai aggiungere CHUNK_SIZE e CHUNK_OVERLAP al tuo core/config.py e file .env se vuoi renderli configurabili, altrimenti usa i default hardcodati).
# Per ogni chunk, copia i metadati del documento/pagina genitore e aggiunge un chunk_sequence_number.
# Tenta di validare i metadati finali del chunk usando DocumentChunkMetadata e model_dump(exclude_none=True) per mantenere i metadati puliti.
# Ottenere Vector Store: Chiama get_vector_store() passando il collection_name specifico, le settings, e l'istanza embeddings già caricata.
# Aggiungere Documenti: Chiama vector_store.add_documents() per indicizzare i chunk.

import os
import shutil
import traceback # Importato
import uuid # Importato
from typing import List, Dict, Any, Optional

from langchain_core.documents import Document as LangchainDocument
from langchain.text_splitter import RecursiveCharacterTextSplitter
from unstructured.partition.auto import partition as unstructured_partition

from models.pydantic_models import DocumentChunkMetadata #, ExtractedLLMMetadata (non usata direttamente qui per ora)
from core.config import Settings
from core.dependencies import get_vector_store # Solo get_vector_store, gli altri sono passati come argomenti
from langchain_huggingface import HuggingFaceEmbeddings # Per type hinting
# from langchain_openai import ChatOpenAI # Per type hinting dell'LLM di tagging


async def process_and_embed_pdf(
    pdf_path: str,
    original_filename: str,
    collection_name: str,
    settings: Settings,
    embeddings_model_instance: HuggingFaceEmbeddings, # Istanza passata
    # llm_for_tagging_instance: ChatOpenAI # Verrà passato quando implementeremo il tagging
) -> int:
    print(f"SERVICE (document_processor): Inizio processamento PDF '{original_filename}' per collection '{collection_name}'.")

    # embeddings_model_instance è già disponibile
    
    print(f"SERVICE: Parsing PDF '{pdf_path}' con unstructured...")
    try:
        raw_elements = unstructured_partition(filename=pdf_path, strategy="auto", languages=['ita', 'eng'])
        print(f"SERVICE: Estratti {len(raw_elements)} elementi da unstructured.")
    except Exception as e_unstructured:
        print(f"SERVICE ERRORE: Fallimento parsing PDF con unstructured: {e_unstructured}")
        raise Exception(f"Errore parsing PDF: {e_unstructured}") from e_unstructured

    documents_to_chunk: List[LangchainDocument] = []
    current_page_number = 1
    page_texts = []
    for el in raw_elements:
        if hasattr(el, 'metadata') and hasattr(el.metadata, 'page_number'):
            current_page_number = el.metadata.page_number
        if hasattr(el, 'text') and el.text and el.text.strip():
            page_texts.append(el.text)

    full_document_text = "\n\n".join(page_texts)
    if not full_document_text.strip():
        print("SERVICE AVVISO: Nessun testo valido estratto dal PDF dopo il join degli elementi.")
        raise ValueError("Nessun contenuto testuale valido estratto dal PDF.")

    base_doc_metadata_dict = {
        "source_filename": original_filename,
        "collection_membership": collection_name,
        "original_page_number": None, # Placeholder
        "inferred_title": None, 
        "summary_brief": None,
        "document_category_llm": None,
        "key_entities": [],
        "keywords_llm": [],
        "detected_language": None
    }
    
    # Validiamo i metadati di base con Pydantic (anche se molti sono None per ora)
    try:
        # Assicuriamoci che tutti i campi richiesti da DocumentChunkMetadata (se ce ne sono)
        # siano presenti in base_doc_metadata_dict o abbiano un default nel modello Pydantic.
        # source_filename e collection_membership sono i più importanti da avere qui.
        # chunk_id verrà aggiunto dopo.
        validated_base_metadata = DocumentChunkMetadata(**base_doc_metadata_dict).model_dump(exclude_none=True)
    except Exception as e_pydantic_base:
        print(f"SERVICE AVVISO: Validazione Pydantic fallita per metadati base: {base_doc_metadata_dict}. Errore: {e_pydantic_base}. Uso metadati non validati.")
        validated_base_metadata = base_doc_metadata_dict


    documents_to_chunk.append(LangchainDocument(page_content=full_document_text, metadata=validated_base_metadata))
    print(f"SERVICE: Creato 1 documento iniziale per il chunking con metadati: {validated_base_metadata}")

    # --- (FASE FUTURA) Integrazione MetadataTagger qui ---
    # ...
    # ----------------------------------------------------

    final_chunks_for_vectorstore: List[LangchainDocument] = []
    chunk_size = getattr(settings, "CHUNK_SIZE", 1000)
    chunk_overlap = getattr(settings, "CHUNK_OVERLAP", 200)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    
    chunk_seq_counter = 0
    for i, doc_to_split in enumerate(documents_to_chunk):
        page_content_to_split = doc_to_split.page_content
        parent_metadata = doc_to_split.metadata.copy()

        splits = text_splitter.split_text(page_content_to_split)
        print(f"SERVICE: Documento/Segmento {i} splittato in {len(splits)} chunks (size: {chunk_size}, overlap: {chunk_overlap}).")

        for split_content in splits:
            chunk_seq_counter += 1
            chunk_specific_metadata = {
                "chunk_sequence_number": chunk_seq_counter,
                "chunk_id": str(uuid.uuid4()) # Aggiunto chunk_id
            }
            final_metadata_for_chunk = {**parent_metadata, **chunk_specific_metadata}
            
            try:
                validated_metadata_chunk = DocumentChunkMetadata(**final_metadata_for_chunk).model_dump(exclude_none=True)
            except Exception as e_pydantic_chunk:
                print(f"SERVICE AVVISO: Validazione Pydantic fallita per metadati chunk: {final_metadata_for_chunk}. Errore: {e_pydantic_chunk}. Uso metadati non validati.")
                validated_metadata_chunk = final_metadata_for_chunk

            chunk_doc = LangchainDocument(page_content=split_content, metadata=validated_metadata_chunk)
            final_chunks_for_vectorstore.append(chunk_doc)
            
    if not final_chunks_for_vectorstore:
        print("SERVICE AVVISO: Nessun chunk generato dopo lo splitting del testo.")
        return 0

    print(f"SERVICE: Numero totale di chunk finali pronti per l'indicizzazione: {len(final_chunks_for_vectorstore)}.")
    if final_chunks_for_vectorstore:
         print(f"SERVICE DEBUG: Esempio metadati primo chunk: {final_chunks_for_vectorstore[0].metadata}")

    print(f"SERVICE: Ottenimento Vector Store per collection '{collection_name}'...")
    try:
        vector_store = get_vector_store(
            collection_name=collection_name,
            settings_dep=settings, # Passa l'oggetto settings
            embeddings_dep=embeddings_model_instance # Passa l'istanza del modello di embedding
        )
    except Exception as e_vs_init:
        print(f"SERVICE ERRORE: Fallimento ottenimento Vector Store per collection '{collection_name}': {e_vs_init}")
        raise Exception(f"Errore Vector Store: {e_vs_init}") from e_vs_init

    print(f"SERVICE: Tentativo di aggiungere {len(final_chunks_for_vectorstore)} documenti alla collection '{collection_name}' nel Vector Store...")
    try:
        vector_store.add_documents(documents=final_chunks_for_vectorstore, ids=None)
        print(f"SERVICE: Aggiunti {len(final_chunks_for_vectorstore)} chunk alla collection '{collection_name}'.")
        return len(final_chunks_for_vectorstore)
    except Exception as e_add_docs:
        print(f"SERVICE ERRORE: Fallimento aggiunta documenti al Vector Store per collection '{collection_name}': {e_add_docs}")
        traceback.print_exc()
        raise Exception(f"Errore durante l'indicizzazione dei documenti: {e_add_docs}") from e_add_docs