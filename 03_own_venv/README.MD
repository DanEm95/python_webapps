# AI Dashboard App

> ⚠️ **WICHTIG:** Zuerst die Datei `example.env.txt` mit den korrekten API-Schlüsseln ausfüllen. Die `.env`-Datei wird daraus anschließend automatisch erstellt und gespeichert.

Einfacher Einstieg, um dein AI-Dashboard (Backend + Frontend) lokal unter Windows 11 zu starten.

---

## 🚀 Schnellstart

1. **`start.bat` ausführen**  
   Doppelklicke auf `start.bat`.  
   Diese Datei übernimmt automatisch:
   - Erstellen der virtuellen Umgebung  
   - Installation aller Abhängigkeiten  
   - Setup der `.env`-Datei (`setup_env.py`)  
   - Start von Backend und Frontend in zwei Terminals  


---

- **OPENAI_API_KEY**: OpenAI GPT-API-Schlüssel ([generieren](https://platform.openai.com/account/api-keys))  
- **S3_KEY / S3_SECRET**: IAM-Zugangsdaten für deinen AWS-S3-Bucket  
- **S3_BUCKET**: Name deines Buckets (z. B. `myawslearninglinkedin`)  
- **S3_REGION**: Region deines Buckets (z. B. `us-east-2`)  
- **S3_PATH**: Optionaler Pfad im Bucket (kann leer bleiben)  
- **MONGO_URL**: MongoDB Atlas Cluster-Verbindungsstring  

---

## 📹 Kurzanleitungsvideos

1. **OpenAI API-Key erstellen (40 Sek.)**  
   ▶️ https://www.youtube.com/watch?v=TdNSj_qgdFk  

2. **AWS IAM-User mit S3-Full-Access anlegen**  
   ▶️ https://www.youtube.com/watch?v=KngM5bfpttA  

3. **MongoDB Atlas Cluster aufsetzen (2 Min.)**  
   ▶️ https://www.youtube.com/watch?v=6stzFjKz1UE  

---

## 📝 API-Beispiele

### FastAPI Call (JSON-Body)
```json
{
  "user_input": "Antworte auf deutsch und sage mir, ob Freitag um 15:00 Uhr das Büro im Café geöffnet hat.",
  "data_source": "s3://myawslearninglinkedin/Landon Hotel Employee Manual_10_25_2023_LIL_34022.docx"
}
```

#### Beispiel-Prompts
- **Prompt 1**  
  > Darf ich mir neben meiner Tätigkeit als Developer als Barkeeper etwas dazuverdienen?

- **Prompt 2**  
  > Du bist Professor an einer Eliteuniversität im MBA-Programm. Schreibe einen prägnanten Blogbeitrag, wie man die „Art of War“ auf moderne Unternehmensstrategien anwendet, mit Beispielen zeitgenössischer Firmen/Unternehmen.

---

## 🔗 Wichtige Links

- **GitHub Repo**  
  https://github.com/DanEm95/python_webapps/tree/64daec26a494cd37adc29f422bb5b44c826cbfe0/03_own_venv

- **AWS S3 Bucket “myawslearninglinkedin”**  
  https://us-east-2.console.aws.amazon.com/s3/buckets/myawslearninglinkedin?region=us-east-2&bucketType=general&tab=objects

- **AWS IAM User “superadmin”**  
  https://us-east-1.console.aws.amazon.com/iam/home?region=us-east-2#/users/details/superadmin?section=security_credentials

- **MongoDB Cloud (Atlas Overview)**  
  https://cloud.mongodb.com/v2/680de52595827e78e61880bd#/overview

---

Viel Erfolg mit dem AI Dashboard! 🚀  