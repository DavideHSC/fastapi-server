# services/__init__.py

# Spiegazione:
# Rendendo process_and_embed_pdf (e in futuro altre funzioni) importabile direttamente da services, semplifichiamo leggermente gli import negli altri file (es. nei router).
# Per ora, ho commentato gli import per rag_service e metadata_extractor perché non li abbiamo ancora creati

# Questo rende le funzioni/classi importabili direttamente da 'from services import ...'
# In alternativa, puoi importare i moduli specifici: from services.document_processor import ...

# Importa le funzioni chiave dai moduli del servizio per renderle accessibili
# più facilmente quando si importa il pacchetto 'services'.
from .document_processor import process_and_embed_pdf
# from .rag_service import get_rag_response # Lo aggiungeremo quando rag_service.py sarà definito
# from .metadata_extractor import extract_metadata_with_llm # Lo aggiungeremo quando metadata_extractor.py sarà definito

# Puoi anche definire un __all__ per specificare cosa viene importato con 'from services import *'
# (generalmente sconsigliato per 'import *')
# __all__ = ['process_and_embed_pdf', 'get_rag_response']

print("DEBUG: Pacchetto 'services' inizializzato (__init__.py).")