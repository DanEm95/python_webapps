Amazon S3 Buckets myawslearninglinkedin:
https://us-east-2.console.aws.amazon.com/s3/buckets/myawslearninglinkedin?region=us-east-2&bucketType=general&tab=objects

MongoDB in Coud:
https://cloud.mongodb.com/v2/680de52595827e78e61880bd#/overview





python -m venv .venv
.venv\Scripts\activate
python.exe -m pip install --upgrade pip
pip install -r requirements.txt --upgrade


fastApi calls "Chat" JSON Body:

{
  "user_input": "Antworte auf deutsch und sage mir ob Freitag um 15:00 Uhr Das Büro im Cafe geöffnet hat.",
  "data_source": "s3://myawslearninglinkedin/Landon Hotel Employee Manual_10_25_2023_LIL_34022.docx"
}