import os
from dotenv import load_dotenv
from pathlib import Path
import shutil

# Hole absoluten Pfad zum aktuellen Skriptverzeichnis
BASE_DIR = Path(__file__).resolve().parent
ENV_PATH = BASE_DIR / ".env"
EXAMPLE_PATH = BASE_DIR / "example.env.txt"

REQUIRED_VARS = [
    "OPENAI_API_KEY",
    "S3_KEY",
    "S3_SECRET",
    "S3_BUCKET",
    "S3_REGION",
    "MONGO_URL"
]

def create_env_from_example():
    if not ENV_PATH.exists():
        if EXAMPLE_PATH.exists():
            shutil.copy(EXAMPLE_PATH, ENV_PATH)
            print("✅ .env wurde aus example.env.txt erstellt.")
        else:
            print("❌ example.env.txt nicht gefunden – bitte manuell erstellen.")
            return False
    return True

def check_env_variables():
    load_dotenv(dotenv_path=ENV_PATH)
    missing = []
    for var in REQUIRED_VARS:
        value = os.getenv(var)
        if not value or value.strip() == "":
            missing.append(var)

    if missing:
        print("⚠️  Folgende Umgebungsvariablen fehlen oder sind leer:")
        for var in missing:
            print(f" - {var}")
        print("➡️  Bitte trage die fehlenden Werte in die .env-Datei ein.")
        return False
    else:
        print("✅ Alle Umgebungsvariablen vorhanden.")
        return True

if __name__ == "__main__":
    print("📦 Setup .env Datei & Validierung")
    if create_env_from_example():
        check_env_variables()
