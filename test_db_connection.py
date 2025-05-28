import psycopg2
import os
from dotenv import load_dotenv

load_dotenv() # Carica le variabili dal tuo file .env

db_host = os.getenv("DB_HOST")
db_port = os.getenv("DB_PORT")
db_name = os.getenv("DB_NAME")
db_user = os.getenv("DB_USER")
db_password = os.getenv("DB_PASSWORD")

print(f"Tentativo di connessione a: Host={db_host}, Port={db_port}, DB={db_name}, User={db_user}")

try:
    conn = psycopg2.connect(
        host=db_host,
        port=db_port,
        dbname=db_name,
        user=db_user,
        password=db_password
    )
    print("Connessione al database PostgreSQL riuscita!")
    conn.close()
except psycopg2.OperationalError as e:
    print(f"Errore di connessione al database: {e}")
    print("Possibili cause:")
    print(f"- Il server PostgreSQL non è in esecuzione su {db_host}:{db_port}.")
    print("- Il firewall sul server Linux (192.168.1.200) sta bloccando la porta 5432.")
    print("- L'indirizzo IP del server o la porta sono errati nel file .env.")
    print("- Le credenziali (utente/password/nome database) sono errate.")
except Exception as e:
    print(f"Si è verificato un altro errore: {e}")