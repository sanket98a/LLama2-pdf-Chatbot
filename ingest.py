from langchain.embeddings import HuggingFaceEmbeddings,HuggingFaceInstructEmbeddings
# from langchain.vectorstores import FAISS
from langchain.document_loaders import PyPDFLoader, DirectoryLoader,CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter 
from langchain.vectorstores import Chroma
import torch
import logging
from constant import (
    CHROMA_SETTINGS,
    PERSIST_DIRECTORY,
    SOURCE_DIRECTORY,
    EMBEDDING_MODEL_NAME
)
if torch.cuda.is_available():
    device = "cuda:0"
else:
    device = "cpu"

device = torch.device(device)
# Create vector database
def create_vector_db():
    # load the data
    path=SOURCE_DIRECTORY.split('\\')[-1]
    print(path)
    loader = DirectoryLoader(SOURCE_DIRECTORY,
                             glob='*.pdf',
                             loader_cls=PyPDFLoader)
    # loader=CSVLoader(file_path='data/review_english_cleaned.csv',encoding="utf8")

    documents = loader.load()
    # print(documents)
    # chunck
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,
                                                   chunk_overlap=50)
    texts = text_splitter.split_documents(documents)
    logging.info(f"Loaded {len(documents)} documents from {SOURCE_DIRECTORY}")
    logging.info(f"Split into {len(texts)} chunks of text")
    
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME,
                                       model_kwargs={'device': device}) 
    
    # embeddings = HuggingFaceInstructEmbeddings(
    #     model_name=EMBEDDING_MODEL_NAME,
    #     model_kwargs={"device": 'cpu'},
    # )

    db = Chroma.from_documents(
        texts,
        embeddings,
        persist_directory=PERSIST_DIRECTORY,
        client_settings=CHROMA_SETTINGS,
    )
    db.persist()
    db = None

if __name__ == "__main__":
    create_vector_db()

