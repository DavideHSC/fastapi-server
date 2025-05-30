# Modifiche e Spiegazioni della Versione "Definitiva per Ora":
# Caching delle Istanze PGVector (_vector_store_cache):
# Ho introdotto un semplice dizionario a livello di modulo _vector_store_cache per cachare le istanze di PGVector basate su una chiave combinata di collection_name e DATABASE_URL. Questo significa che se richiedi lo store per la stessa collection più volte (anche in richieste diverse, ma nello stesso processo worker), non verrà ricreato da zero ogni volta, migliorando l'efficienza.
# La chiave cache_key è stata resa un po' più robusta.
# get_vector_store:
# Ora utilizza la cache _vector_store_cache.
# Usa settings_dep.DATABASE_URL (es. postgresql://...) come stringa di connessione, che è la forma preferita da langchain-postgres.
# Ho mantenuto i commenti su use_jsonb=True (raccomandato) e pre_delete_collection=False (per produzione).
# get_rag_llm:
# Verifica più esplicitamente che OPENROUTER_API_KEY esista e abbia un valore prima di procedere.
# Passa settings_dep.OPENROUTER_API_KEY (che è già un SecretStr o None) direttamente al parametro api_key di ChatOpenAI.
# Ho impostato max_tokens=1024 come esempio; questo valore andrà calibrato.
# get_metadata_tagging_llm (Implementato, non più "futuro"):
# Questa funzione ora è completamente implementata.
# Per ora, usa lo stesso modello LLM (OPENROUTER_MODEL_NAME) e la stessa API key del get_rag_llm. Questo è un punto di partenza ragionevole per non complicare troppo la configurazione iniziale.
# Differenza Chiave: Imposta temperature=0. Per task di estrazione di informazioni e generazione di output strutturato (come il tagging dei metadati), una temperatura bassa o zero è generalmente preferibile per ottenere risultati più consistenti, fattuali e meno "creativi".
# Ha un max_tokens separato (es. 512), dato che l'output dei metadati è generalmente più corto di una risposta RAG completa.
# Scenario Futuro: Se scopri che il tagging richiede un modello diverso (es. uno più piccolo ed economico, o uno specializzato nell'estrazione) o una API key diversa, potresti aggiungere nuove variabili in core/config.py (es. METADATA_LLM_MODEL_NAME, METADATA_LLM_API_KEY) e usare quelle qui. Per adesso, riutilizziamo la configurazione di OpenRouter esistente.
# Commenti: Ho cercato di rendere i commenti più esplicativi dell'uso attuale e delle ragioni dietro certe scelte, piuttosto che etichettare le cose come "future".
# Considerazioni:
# use_jsonb=True per PGVector: Ti consiglio vivamente di abilitare questa opzione (rimuovendo il commento) quando sei pronto. Migliora le capacità di interrogazione dei metadati JSON in PostgreSQL. Se PGVector crea la tabella per te, dovrebbe configurare la colonna dei metadati come JSONB. Se la tabella esiste già con una colonna JSON per i metadati, potrebbe essere necessaria una migrazione manuale del tipo di colonna nel database.
# max_tokens in get_rag_llm: Ho reinserito max_tokens. Se Pylance dovesse ancora lamentarsi specificamente per ChatOpenAI e questo parametro (nonostante sia documentato come valido per i default in LangChain), e a runtime funziona, puoi ignorare l'avviso o, come soluzione alternativa, passarlo tramite model_kwargs={"max_tokens": 1024} nel costruttore di ChatOpenAI.

from typing import Optional, Dict
from functools import lru_cache

from fastapi import HTTPException, Depends
from pydantic import SecretStr

from core.config import Settings, settings as global_settings # Importa l'istanza globale
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_postgres.vectorstores import PGVector
from langchain_openai import ChatOpenAI

@lru_cache()
def get_settings() -> Settings:
    print("DEBUG: Accesso alle configurazioni globali (get_settings).")
    return global_settings

# --- MODIFICA QUI: get_embeddings_model ---
@lru_cache()
def get_embeddings_model() -> HuggingFaceEmbeddings: # Rimosso settings_dep come argomento
    """
    Inizializza e restituisce un'istanza cachata del modello di embedding.
    Accede alle configurazioni globali tramite get_settings().
    """
    current_settings = get_settings() # Chiama la dipendenza cachata per le impostazioni
    print(f"DEBUG: Inizializzazione/Recupero modello embedding: '{current_settings.EMBEDDING_MODEL_NAME}'.")
    try:
        model = HuggingFaceEmbeddings(
            model_name=current_settings.EMBEDDING_MODEL_NAME,
            model_kwargs={'device': 'cpu'}
        )
        print(f"DEBUG: Modello embedding '{current_settings.EMBEDDING_MODEL_NAME}' pronto.")
        return model
    except Exception as e:
        print(f"ERRORE CRITICO: Fallimento caricamento modello embedding '{current_settings.EMBEDDING_MODEL_NAME}': {e}")
        raise HTTPException(status_code=503, detail=f"Errore inizializzazione modello di embedding: {e}")
# --- FINE MODIFICA ---

_vector_store_cache: Dict[str, PGVector] = {}

def get_vector_store(
    collection_name: str,
    settings_dep: Settings = Depends(get_settings), # settings_dep è ancora necessario qui per DATABASE_URL
    embeddings_dep: HuggingFaceEmbeddings = Depends(get_embeddings_model) # Ora get_embeddings_model non prende settings_dep
) -> PGVector:
    print(f"DEBUG: Chiamata a get_vector_store per collection '{collection_name}'.")
    if not embeddings_dep:
        print("ERRORE FATALE: Modello di embedding non disponibile per get_vector_store.")
        raise HTTPException(status_code=503, detail="Modello di embedding non inizializzato correttamente.")

    cache_key = f"{collection_name}_{settings_dep.DATABASE_URL}" # settings_dep qui è OK perché è l'oggetto Settings globale
    if cache_key in _vector_store_cache:
        print(f"DEBUG: Riutilizzo istanza PGVector cachata per collection '{collection_name}'.")
        return _vector_store_cache[cache_key]
    
    print(f"DEBUG: Creazione nuova istanza PGVector per collection '{collection_name}'.")
    try:
        vector_store = PGVector(
            embeddings=embeddings_dep,
            collection_name=collection_name,
            connection=settings_dep.DATABASE_URL,
        )
        print(f"DEBUG: Istanza PGVector per collection '{collection_name}' creata con successo.")
        _vector_store_cache[cache_key] = vector_store
        return vector_store
    except Exception as e:
        print(f"ERRORE CRITICO: Fallimento creazione/connessione PGVector store per collection '{collection_name}': {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=503, detail=f"Impossibile creare/connettersi al Vector Store '{collection_name}': {e}")

# get_rag_llm e get_metadata_tagging_llm rimangono invariati (usano Depends(get_settings))
@lru_cache()
def get_rag_llm(settings_dep: Settings = Depends(get_settings)) -> ChatOpenAI:
    # ... (codice invariato) ...
    print(f"DEBUG: Inizializzazione/Recupero LLM per RAG: Modello '{settings_dep.OPENROUTER_MODEL_NAME}'.")
    if not settings_dep.OPENROUTER_API_KEY or not settings_dep.OPENROUTER_API_KEY.get_secret_value():
        print("ERRORE CRITICO: OPENROUTER_API_KEY non configurata per RAG LLM.")
        raise HTTPException(status_code=500, detail="Configurazione LLM API Key (OPENROUTER_API_KEY) mancante.")
    try:
        llm = ChatOpenAI(
            model=settings_dep.OPENROUTER_MODEL_NAME.strip('"'),
            api_key=settings_dep.OPENROUTER_API_KEY,
            base_url=settings_dep.OPENROUTER_API_BASE,
            temperature=0.1,
            max_tokens=1024
        )
        print("DEBUG: LLM per RAG inizializzato/recuperato con successo.")
        return llm
    except Exception as e:
        print(f"ERRORE CRITICO: Inizializzazione LLM per RAG fallita: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=503, detail=f"Errore durante l'inizializzazione dell'LLM per RAG: {e}")

@lru_cache()
def get_metadata_tagging_llm(settings_dep: Settings = Depends(get_settings)) -> ChatOpenAI:
    # ... (codice invariato) ...
    print(f"DEBUG: Inizializzazione/Recupero LLM per Metadata Tagging: Modello '{settings_dep.OPENROUTER_MODEL_NAME}'.")
    if not settings_dep.OPENROUTER_API_KEY or not settings_dep.OPENROUTER_API_KEY.get_secret_value():
        print("ERRORE CRITICO: OPENROUTER_API_KEY non configurata per Metadata Tagging LLM.")
        raise HTTPException(status_code=500, detail="Configurazione LLM API Key (OPENROUTER_API_KEY) mancante per il tagging.")
    try:
        tagging_llm = ChatOpenAI(
            model=settings_dep.OPENROUTER_MODEL_NAME.strip('"'),
            api_key=settings_dep.OPENROUTER_API_KEY,
            base_url=settings_dep.OPENROUTER_API_BASE,
            temperature=0,
            max_tokens=512
        )
        print("DEBUG: LLM per Metadata Tagging inizializzato/recuperato con successo.")
        return tagging_llm
    except Exception as e:
        print(f"ERRORE CRITICO: Inizializzazione LLM per Metadata Tagging fallita: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=503, detail=f"Errore durante l'inizializzazione dell'LLM per il tagging: {e}")