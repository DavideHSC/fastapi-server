import os
import requests
import json
from dotenv import load_dotenv

# Carica le variabili d'ambiente dal file .env
load_dotenv(override=True)

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_API_BASE = os.getenv("OPENROUTER_API_BASE", "https://openrouter.ai/api/v1") # Usa il default se non nel .env
OPENROUTER_MODEL_NAME = os.getenv("OPENROUTER_MODEL_NAME", "qwen/qwen3-235b-a22b:free").strip('"')

# Verifica che la chiave API sia stata caricata
if not OPENROUTER_API_KEY or OPENROUTER_API_KEY == "fallback_apikey":
    print("ERRORE: La variabile d'ambiente OPENROUTER_API_KEY non è impostata correttamente nel file .env o ha il valore di fallback.")
    exit()

print(f"--- Test Chiamata Diretta a OpenRouter ---")
print(f"API Key (prime 10 char): {OPENROUTER_API_KEY[:10]}...")
print(f"API Base URL: {OPENROUTER_API_BASE}")
print(f"Modello: {OPENROUTER_MODEL_NAME}")
print("-" * 40)

# Endpoint per le completions della chat
url = f"{OPENROUTER_API_BASE}/chat/completions"

headers = {
    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
    "Content-Type": "application/json",
    # OpenRouter potrebbe richiedere questo header per reindirizzare le richieste al modello corretto
    # se l'URL base è generico. Includiamolo per sicurezza, anche se non sempre strettamente necessario.
    "HTTP-Referer": "http://localhost", # Può essere un valore fittizio, alcuni modelli lo usano per il routing
    "X-Title": "Test FastAPI App"      # Anche questo può essere fittizio
}

data = {
    "model": OPENROUTER_MODEL_NAME,
    "messages": [
        {"role": "user", "content": "Ciao, qual è la capitale della Francia?"}
    ],
    "temperature": 0.1,
    "max_tokens": 50
}

try:
    print(f"Invio richiesta a: {url}")
    print(f"Headers: {headers}") # Stampa gli header per debug (esclusa la chiave completa)
    print(f"Data (payload): {json.dumps(data, indent=2)}")
    
    response = requests.post(url, headers=headers, json=data, timeout=30) # Timeout di 30 secondi

    print("-" * 40)
    print(f"Status Code della Risposta: {response.status_code}")
    
    try:
        response_json = response.json()
        print("Risposta JSON:")
        print(json.dumps(response_json, indent=2))
    except json.JSONDecodeError:
        print("Corpo della Risposta (non JSON):")
        print(response.text)

    if response.status_code == 401:
        print("\nERRORE 401: Autenticazione fallita!")
        print("Possibili cause:")
        print("1. La OPENROUTER_API_KEY nel file .env è errata, invalida, o non ha permessi.")
        print("2. Hai copiato male la chiave (spazi extra, caratteri mancanti).")
        print("3. Problemi con il tuo account OpenRouter (credito, limiti, chiave revocata).")
        print("Verifica la tua API key sul dashboard di OpenRouter.")
    elif response.status_code == 200:
        print("\nSUCCESSO! La chiamata API a OpenRouter è andata a buon fine.")
    else:
        print(f"\nERRORE {response.status_code}: La chiamata API è fallita per un altro motivo.")

except requests.exceptions.RequestException as e:
    print(f"\nERRORE di Connessione/Richiesta: {e}")
    print("Possibili cause:")
    print("1. Problemi di rete (nessuna connessione internet, firewall).")
    print(f"2. L'URL base '{OPENROUTER_API_BASE}' è errato o il servizio non è raggiungibile.")
    print("3. Timeout della richiesta.")