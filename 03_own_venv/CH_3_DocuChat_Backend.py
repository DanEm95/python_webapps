#TODO datein sollen auf MongoDB hochgeladen werden.

from pydantic import BaseModel
import pymongo
import os
import sys
import traceback
import logging
from fastapi import FastAPI, UploadFile, HTTPException, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import Docx2txtLoader, PyPDFLoader
from langchain_community.callbacks import get_openai_callback
from langchain.chains import ConversationalRetrievalChain
from langchain_openai import ChatOpenAI
import gc
import uuid
from typing import List
import awswrangler as wr
import boto3
from dotenv import load_dotenv
import pymongo
import os
import shutil  # Import shutil

# Setup logging
logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

S3_KEY = os.environ.get("S3_KEY")
S3_SECRET = os.environ.get("S3_SECRET")
S3_BUCKET = os.environ.get("S3_BUCKET")
S3_REGION = os.environ.get("S3_REGION")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
MONGO_URL = os.environ.get("MONGO_URL")
S3_PATH = os.environ.get("S3_PATH")

# MongoDB connection
try:
    MONGO_URL = "mongodb+srv://admin:admin@cluster0.wdbur6s.mongodb.net/?retryWrites=true&w=majority&appName=cluster0"
    client = pymongo.MongoClient(MONGO_URL, uuidRepresentation="standard")
    db = client["chat_with_doc"]
    conversationcol = db["chat-history"]
    conversationcol.create_index([("session_id")], unique=True)
except Exception:
    logger.error("MongoDB connection error: %s", traceback.format_exc())
    exc_type, exc_obj, exc_tb = sys.exc_info()
    fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
    logger.error(f"Error at {fname} line {exc_tb.tb_lineno}: {exc_type}")
    raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Database connection failed")

# AWS S3 session
aws_s3 = boto3.Session(
    aws_access_key_id=S3_KEY,
    aws_secret_access_key=S3_SECRET,
    region_name=S3_REGION,
)

# Pydantic model for chat message
class ChatMessageSent(BaseModel):
    session_id: str = None
    user_input: str
    data_source: str

# Function to get a response from GPT
def get_response(file_name: str, session_id: str, query: str, model: str = "gpt-4.1-nano", temperature: float = 0):
    logger.info("Processing file: %s", file_name)
    file_name = os.path.basename(file_name)  # Extract file name from the path

    embeddings = OpenAIEmbeddings()
    local_file_path = os.path.join("temp", file_name)

    try:

        if not os.path.exists(local_file_path):
            logger.info(f"Downloading {file_name} from S3...")
            wr.s3.download(path=f"s3://{S3_BUCKET}/{S3_PATH}{file_name}", local_file=local_file_path, boto3_session=aws_s3)
        else:
            logger.info(f"{file_name} already exists locally, skipping download.")


        # Load data - CORRECT PATH
        if file_name.endswith(".docx"):
            loader = Docx2txtLoader(file_path=local_file_path)
        else:
            loader = PyPDFLoader(file_path=local_file_path)

        data = loader.load()

        # Split data for token limits
        logger.info("Splitting text into chunks")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=0, separators=["\n", " ", ""]
        )
        all_splits = text_splitter.split_documents(data)

        # Store data in vector database
        vectorstore = FAISS.from_documents(all_splits, embeddings)

        # Initialize OpenAI
        llm = ChatOpenAI(model_name=model, temperature=temperature)

        # Use ConversationalRetrievalChain
        qa_chain = ConversationalRetrievalChain.from_llm(
            llm, retriever=vectorstore.as_retriever()
        )

        with get_openai_callback() as cb:
            try:
                # Use 'invoke' instead of '__call__'
                answer = qa_chain.invoke(
                    {
                        "question": query,
                        "chat_history": load_memory_to_pass(session_id=session_id),
                    }
                )
                logger.info(f"Total Tokens: {cb.total_tokens}")
                logger.info(f"Prompt Tokens: {cb.prompt_tokens}")
                logger.info(f"Completion Tokens: {cb.completion_tokens}")
                logger.info(f"Total Cost (USD): ${cb.total_cost}")
                answer["total_tokens_used"] = cb.total_tokens
            except Exception as e:
                logger.error("Error during OpenAI API call: %s", str(e))
                raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="OpenAI API error")

        gc.collect()  # Collect garbage

        return answer
    except Exception as e:
        logger.error(f"Error during get_response: {e}")
        raise e


# Function to load conversation history
def load_memory_to_pass(session_id: str):
    data = conversationcol.find_one({"session_id": session_id})
    history = []
    if data:
        data = data["conversation"]
        for x in range(0, len(data), 2):
            history.append((data[x], data[x + 1]))
    logger.info("Loaded chat history: %s", history)
    return history

# Function to generate session ID
def get_session() -> str:
    return str(uuid.uuid4())

# Function to add session history
def add_session_history(session_id: str, new_values: List):
    document = conversationcol.find_one({"session_id": session_id})
    if document:
        conversation = document["conversation"]
        conversation.extend(new_values)
        conversationcol.update_one(
            {"session_id": session_id}, {"$set": {"conversation": conversation}}
        )
    else:
        conversationcol.insert_one(
            {"session_id": session_id, "conversation": new_values}
        )

# FastAPI app
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Endpoint to handle chat messages
@app.post("/chat")
async def create_chat_message(chats: ChatMessageSent):
    try:
        if chats.session_id is None:
            session_id = get_session()
            payload = ChatMessageSent(
                session_id=session_id,
                user_input=chats.user_input,
                data_source=chats.data_source,
            )
            response = get_response(
                file_name=payload.data_source,  # Use direct attribute access here
                session_id=payload.session_id,  # Use direct attribute access here
                query=payload.user_input,  # Use direct attribute access here
            )
            add_session_history(session_id=session_id, new_values=[payload.user_input, response["answer"]])
            return JSONResponse(content={"response": response, "session_id": str(session_id)})

        else:
            payload = ChatMessageSent(
                session_id=str(chats.session_id),
                user_input=chats.user_input,
                data_source=chats.data_source,
            )
            response = get_response(
                file_name=payload.data_source,  # Use direct attribute access here
                session_id=payload.session_id,  # Use direct attribute access here
                query=payload.user_input,  # Use direct attribute access here
            )
            add_session_history(session_id=str(chats.session_id), new_values=[payload.user_input, response["answer"]])
            return JSONResponse(content={"response": response, "session_id": str(chats.session_id)})

    except Exception:
        logger.error("Error during chat message processing: %s", traceback.format_exc())
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        logger.error(f"Error at {fname} line {exc_tb.tb_lineno}: {exc_type}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Unexpected error")

# Endpoint to upload file to S3
import shutil  # Import shutil

@app.post("/uploadFile")
async def uploadtos3(data_file: UploadFile):
    print("Received file upload request")  # Log to the console
    target_dir = "temp"  # Save directly to "temp"
    file_path = os.path.join(target_dir, data_file.filename)

    try:
        os.makedirs(target_dir, exist_ok=True) # Ensure temp folder exists

        with open(file_path, "wb") as out_file:
            content = await data_file.read()
            out_file.write(content)

        print(f"File {file_path} saved successfully.")
        response = {
            "filename": data_file.filename,
            "file_path": data_file.filename #Just file name
        }
        return JSONResponse(content=response)

    except Exception as e:
        print(f"Error uploading file: {e}")
        raise HTTPException(status_code=500, detail="Error uploading file")

# Run the app
import uvicorn

if __name__ == "__main__":
    uvicorn.run(app)
