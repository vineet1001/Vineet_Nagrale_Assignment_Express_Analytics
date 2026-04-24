import os
import bs4
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma

def build_vector_store():
    urls = [
        "https://pytorch.org/tutorials/beginner/basics/tensorqs_tutorial.html",
        "https://pytorch.org/tutorials/beginner/basics/buildmodel_tutorial.html",
        "https://pytorch.org/tutorials/beginner/basics/autogradqs_tutorial.html"
    ]

    print("Loading documents from web...")
    
    loader = WebBaseLoader(web_paths=urls)
    docs = loader.load()

    print("Chunking documents...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True 
    )
    
    splits = text_splitter.split_documents(docs)

    print("Generating embeddings and indexing into ChromaDB...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    vectorstore = Chroma.from_documents(
        documents=splits, 
        embedding=embeddings, 
        persist_directory="./chroma_db" 
    )

    print("Ingestion complete! Vector database saved to ./chroma_db")

if __name__ == "__main__":
    build_vector_store()