from pydantic import SecretStr, PostgresDsn, Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional, List

class Settings(BaseSettings):
    """
    Configurazioni dell'applicazione caricate da variabili d'ambiente e file .env.
    """
    APP_NAME: str = "Knowledge Base API" # O leggilo da .e
    
    model_config = SettingsConfigDict(
        env_file='.env',                # Nome del file .env
        env_file_encoding='utf-8',      # Encoding del file .env
        extra='ignore'                  # Ignora variabili d'ambiente extra non definite qui
    )
    
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200
    # USE_LLM_METADATA_TAGGING: bool = False # Per attivare/disattivare il tagging LLM

    # Configurazione Database PGVector
    DB_HOST: str = "localhost"
    DB_PORT: int = 5432
    DB_USER: str = "postgres"
    DB_PASSWORD: SecretStr = SecretStr("password") # Fornisci un default o rendilo obbligatorio
    DB_NAME: str = "mydatabase"
    DEFAULT_COLLECTION_NAME: str = "my_knowledge_docs" # Collection di default se non specificata

    # Configurazione Modello di Embedding
    EMBEDDING_MODEL_NAME: str = "all-MiniLM-L6-v2"

    # Configurazione LLM (OpenRouter)
    OPENROUTER_API_KEY: Optional[SecretStr] = Field(None) # La chiave è opzionale se non si usano funzioni LLM
    OPENROUTER_MODEL_NAME: str = "qwen/qwen2-72b-instruct" # Aggiornato a un modello Qwen2 più comune per OpenRouter
    OPENROUTER_API_BASE: str = "https://openrouter.ai/api/v1"

    # (Futuro) Configurazione LLM per Metadata Tagging (potrebbe essere diverso)
    # METADATA_LLM_MODEL_NAME: str = "gpt-3.5-turbo-0613" # Esempio
    # METADATA_LLM_API_KEY: Optional[SecretStr] = None
    # METADATA_LLM_API_BASE: Optional[str] = None


    @property
    def DATABASE_URL(self) -> str:
        """
        Genera la DSN (Data Source Name) per la connessione al database PostgreSQL
        compatibile con SQLAlchemy e langchain-postgres.
        """
        # langchain-postgres preferisce una DSN senza il driver +psycopg2 specificato
        # psycopg (la libreria usata da langchain-postgres) lo gestirà.
        return f"postgresql://{self.DB_USER}:{self.DB_PASSWORD.get_secret_value()}@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}"

    @property
    def DATABASE_URL_SQLALCHEMY(self) -> str:
        """
        Genera la DSN (Data Source Name) per la connessione al database PostgreSQL
        esplicitamente per SQLAlchemy (se necessario altrove).
        """
        return f"postgresql+psycopg2://{self.DB_USER}:{self.DB_PASSWORD.get_secret_value()}@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}"


# Istanza globale delle configurazioni, accessibile importando 'settings' da questo modulo
settings = Settings() # type: ignore[call-arg]

# Per testare che il caricamento funzioni (puoi commentarlo dopo)
if __name__ == "__main__":
    print("Configurazioni Caricate:")
    print(f"DB Host: {settings.DB_HOST}")
    print(f"DB Port: {settings.DB_PORT}")
    print(f"DB User: {settings.DB_USER}")
    print(f"DB Password: {settings.DB_PASSWORD.get_secret_value()[:3]}*** (solo le prime 3 lettere per sicurezza)") # Non stampare mai la password completa
    print(f"DB Name: {settings.DB_NAME}")
    print(f"Database URL (per langchain-postgres): {settings.DATABASE_URL}")
    print(f"Database URL (SQLAlchemy): {settings.DATABASE_URL_SQLALCHEMY}")
    print(f"Default Collection Name: {settings.DEFAULT_COLLECTION_NAME}")
    print(f"Embedding Model: {settings.EMBEDDING_MODEL_NAME}")
    if settings.OPENROUTER_API_KEY:
        print(f"OpenRouter API Key: Presente (prime 10: {settings.OPENROUTER_API_KEY.get_secret_value()[:10]}...)")
    else:
        print("OpenRouter API Key: Non fornita")
    print(f"OpenRouter Model: {settings.OPENROUTER_MODEL_NAME}")
    print(f"OpenRouter Base URL: {settings.OPENROUTER_API_BASE}")