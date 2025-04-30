@echo off
echo Setup und Start der AI App...

:: Virtuelle Umgebung erstellen (falls nicht vorhanden)
if not exist .venv (
    echo Erstelle virtuelle Umgebung...
    python -m venv .venv
)

:: Aktivieren der Umgebung
call .venv\Scripts\activate.bat

:: Pip aktualisieren + requirements installieren
python -m pip install --upgrade pip
if exist requirements.txt (
    pip install -r requirements.txt --upgrade
)

:: .env Setup prüfen
echo Überprüfe .env-Konfiguration...
python setup_env.py

:: Backend starten (neues Terminalfenster)
start "Backend" cmd /k ".venv\Scripts\activate && python CH_3_DocuChat_Backend.py"

:: Frontend starten (neues Terminalfenster mit Streamlit)
start "Frontend" cmd /k ".venv\Scripts\activate && streamlit run CH_4_DocuChat_Frontend.py"
