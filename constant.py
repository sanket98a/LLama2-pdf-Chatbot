import os

# from dotenv import load_dotenv
from chromadb.config import Settings


# load_dotenv()
ROOT_DIRECTORY = os.path.dirname(os.path.realpath(__file__))

# Define the folder for storing database
SOURCE_DIRECTORY = f"{ROOT_DIRECTORY}/data"

PERSIST_DIRECTORY = f"{ROOT_DIRECTORY}/DB"

# Can be changed to a specific number
INGEST_THREADS = os.cpu_count() or 8

# Define the Chroma settings
CHROMA_SETTINGS = Settings(
    chroma_db_impl="duckdb+parquet", persist_directory=PERSIST_DIRECTORY, anonymized_telemetry=False
)

# HUGGING FACE EMBEDDING MODEL
EMBEDDING_MODEL_NAME="sentence-transformers/all-MiniLM-L6-v2"

# HUGGING FACE INSTRUCT EMBEDDING MODEL
# EMBEDDING_MODEL_NAME="./hkunlp_instructor-large"




