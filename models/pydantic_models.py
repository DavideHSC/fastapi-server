# Modifiche e Spiegazioni:
# UploadResponse:
# Aggiunto collection_used: str per informare il client in quale collection (tabella) è stato indicizzato il documento.
# QueryRequest:
# Aggiunto document_category: Optional[str] con una descrizione. Questo permetterà al frontend (o al client API) di specificare quale knowledge base interrogare. Ho aumentato le=20 per top_k.
# QueryResponseSourceDocument e QueryResponse:
# Ho creato QueryResponseSourceDocument per strutturare meglio i documenti sorgente restituiti. Invece di una semplice lista di stringhe (source_chunks), ora restituiamo una lista di questi oggetti, ognuno contenente page_content e i suoi metadata. Questo dà al frontend più contesto.
# QueryResponse ora usa source_documents: List[QueryResponseSourceDocument].
# Aggiunto collection_queried: str.
# ExtractedLLMMetadata:
# Questo è lo schema chiave per l'estrazione da parte dell'LLM.
# Ho dato nomi ai campi che suggeriscono la loro origine (es. _llm come suffisso o inferred_) per distinguerli da metadati impostati manualmente.
# Le descrizioni dei campi sono cruciali perché l'LLM le userà per capire cosa estrarre. Rendile il più chiare e specifiche possibile.
# Ho usato Optional e default_factory=list dove appropriato.
# document_category_llm usa Literal per suggerire all'LLM un insieme chiuso di categorie, ma l'LLM potrebbe comunque restituire altro.
# Config.extra = "ignore": Se l'LLM restituisce campi non definiti in questo schema, verranno ignorati invece di causare un errore di validazione Pydantic. Questo può essere utile se l'LLM è verboso.
# DocumentChunkMetadata:
# Questo modello rappresenta i metadati che verranno effettivamente salvati insieme a ogni chunk nel vector store.
# Include metadati di base che imposterai tu nel codice (come source_filename, collection_membership, original_page_number).
# Poi "appiattisce" (include direttamente) i campi da ExtractedLLMMetadata. Questo perché molti vector store e sistemi di query funzionano meglio con metadati piatti piuttosto che annidati (es. metadata.llm_extracted.inferred_title vs metadata.inferred_title). Ho reso i campi direttamente opzionali qui, assumendo che se l'LLM non estrae qualcosa, quel campo sarà None.
# Config.extra = "allow": Questo è importante qui. Quando LangChain crea i suoi oggetti Document, potrebbe avere i propri campi interni nei metadati (es. a volte aggiunge un _id o altri). "allow" previene errori di validazione Pydantic se ci sono campi extra.
# Aggiunto chunk_id opzionale.
# CollectionInfo:
# Un piccolo modello che potrebbe essere utile in futuro se vuoi creare un endpoint API che elenca le collection disponibili e alcune informazioni su di esse.
# Considerazioni:
# Specificità vs. Generalità: Lo schema ExtractedLLMMetadata è un compromesso. Se hai tipi di documenti molto diversi, potresti persino avere schemi di metadati diversi per ciascuno e decidere quale schema usare per il tagging LLM in base al document_category fornito dall'utente durante l'upload. Per ora, uno schema unificato è un buon punto di partenza.
# Iterazione: Probabilmente dovrai iterare su ExtractedLLMMetadata dopo aver visto cosa l'LLM riesce effettivamente a estrarre e cosa ti è più utile per le query.

from pydantic import BaseModel, Field, SecretStr # SecretStr potrebbe non servire qui, ma lo tengo per coerenza se altri modelli lo usano
from typing import List, Optional, Literal

# --- Modelli per gli Endpoint API ---

class UploadResponse(BaseModel):
    filename: str
    message: str
    chunks_added: int
    collection_used: str

class QueryRequest(BaseModel):
    question: str
    top_k: int = Field(default=3, ge=1, le=20, description="Numero di frammenti più rilevanti da recuperare.")
    document_category: Optional[str] = Field(
        None, 
        description="Opzionale. Specifica la categoria o il nome della collection da interrogare (es. 'listini', 'manuali_prodotto_a'). Se omesso, verrà usata la collection di default."
    )

class QueryResponseSourceDocument(BaseModel): # Nuovo modello per rappresentare i documenti sorgente
    page_content: str
    metadata: dict # Manteniamo generico per ora, ma potrebbe essere ChunkMetadata

class QueryResponse(BaseModel):
    question: str
    answer: str
    source_documents: List[QueryResponseSourceDocument] # Modificato per restituire oggetti più strutturati
    collection_queried: str

# --- Modelli per l'Estrazione e l'Archiviazione di Metadati ---

class ExtractedLLMMetadata(BaseModel):
    """
    Schema per i metadati strutturati da estrarre dal contenuto di un documento/pagina
    utilizzando un Large Language Model (LLM).
    """
    inferred_title: Optional[str] = Field(
        None, 
        description="Il titolo più probabile, l'oggetto o l'argomento principale conciso della pagina o del segmento di documento analizzato."
    )
    summary_brief: Optional[str] = Field(
        None,
        description="Un riassunto molto breve, una o due frasi al massimo, che catturi l'essenza del contenuto del documento/pagina."
    )
    document_category_llm: Optional[Literal[
        "listino prezzi", 
        "manuale utente/istruzioni", 
        "specifiche tecniche/requisiti", 
        "contratto/termini legali", 
        "report/analisi", 
        "articolo/news", 
        "corrispondenza/email",
        "documento generale/non classificato"
    ]] = Field(
        None, 
        description="La classificazione più appropriata del tipo di contenuto del documento o della pagina."
    )
    key_entities: Optional[List[str]] = Field(
        default_factory=list, 
        description="Lista delle 3-5 entità nominate (es. prodotti, aziende, persone, luoghi, concetti chiave) più importanti menzionate."
    )
    keywords_llm: Optional[List[str]] = Field(
        default_factory=list, 
        description="Lista di 3-7 parole chiave o frasi brevi significative che rappresentano i temi principali del testo."
    )
    detected_language: Optional[str] = Field(
        None, 
        description="La lingua principale rilevata nel testo (es. 'it', 'en', 'fr')."
    )
    # Esempi di altri campi che potresti voler estrarre:
    # product_codes_llm: Optional[List[str]] = Field(default_factory=list, description="Eventuali codici prodotto o SKU rilevati nel testo.")
    # relevant_dates_llm: Optional[List[str]] = Field(default_factory=list, description="Date importanti o periodi temporali menzionati.")
    # sentiment_llm: Optional[Literal["positivo", "negativo", "neutro"]] = Field(None, description="Il sentiment generale espresso nel testo.")

    class Config:
        # Se l'LLM aggiunge campi non definiti qui, Pydantic darà errore per impostazione predefinita.
        # Per l'estrazione di metadati, è generalmente meglio essere restrittivi (`extra='forbid'`)
        # per assicurarsi che l'LLM stia popolando i campi attesi.
        # Se vuoi permettere campi extra: extra = "allow"
        # Se vuoi ignorare campi extra: extra = "ignore"
        extra = "ignore" # Ignora campi extra che l'LLM potrebbe restituire ma non sono nello schema


class DocumentChunkMetadata(BaseModel):
    """
    Metadati finali associati a ogni chunk indicizzato nel vector store.
    Include metadati di base e quelli estratti dall'LLM.
    """
    # Metadati di base, impostati durante il processing
    source_filename: str = Field(description="Nome del file PDF originale.")
    collection_membership: str = Field(description="Nome della collection/tabella a cui appartiene questo chunk.")
    original_page_number: Optional[int] = Field(None, description="Numero di pagina del PDF da cui proviene il chunk, se disponibile.")
    chunk_id: Optional[str] = Field(None, description="Un ID univoco per questo specifico chunk.") # Potrebbe essere generato (uuid)

    # Metadati estratti dall'LLM (ereditati dalla pagina/documento genitore)
    # Usiamo il "flattening" o l'incorporazione diretta dei campi di ExtractedLLMMetadata.
    # In alternativa, potresti avere: llm_extracted: Optional[ExtractedLLMMetadata] = None
    # Ma averli piatti può essere più facile per alcune query/filtri nel vector store.
    inferred_title: Optional[str] = None
    summary_brief: Optional[str] = None
    document_category_llm: Optional[str] = None # Stringa per flessibilità, anche se l'LLM dovrebbe restituire dal Literal
    key_entities: List[str] = Field(default_factory=list)
    keywords_llm: List[str] = Field(default_factory=list)
    detected_language: Optional[str] = None
    # Aggiungi qui i campi opzionali da ExtractedLLMMetadata che vuoi usare
    # product_codes_llm: List[str] = Field(default_factory=list)
    # relevant_dates_llm: List[str] = Field(default_factory=list)
    # sentiment_llm: Optional[str] = None


    class Config:
        # Permetti la validazione quando si assegnano valori ai modelli
        validate_assignment = True
        # Se hai altri campi nei metadati dei documenti Langchain che vuoi mantenere:
        extra = "allow" # Permetti campi extra che potrebbero venire da Langchain Document.metadata


# Modello per il frontend se vuoi mostrare una lista di collection disponibili
class CollectionInfo(BaseModel):
    name: str
    document_count: Optional[int] = None
    description: Optional[str] = None