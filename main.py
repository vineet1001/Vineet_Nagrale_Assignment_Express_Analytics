from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import os


from graph import app as langgraph_app

from ingestion import build_vector_store

from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

app = FastAPI(
    title="Technical Docs RAG Assistant",
    description="A LangGraph-powered self-corrective RAG API",
    version="1.0.0"
)


class QueryRequest(BaseModel):
    question: str

class FeedbackRequest(BaseModel):
    feedback: str 
    comment: Optional[str] = None



@app.post("/query", summary="Submit a question")
async def query_assistant(request: QueryRequest):
    """Submits a natural language question to the LangGraph workflow."""
    try:
        inputs = {"question": request.question, "retries": 0}
        
        result = langgraph_app.invoke(inputs)
        
        
        sources = []
        if result.get("documents"):
            for doc in result["documents"]:
                source = doc.metadata.get("source")
                if source:
                    sources.append(source)
                    
        return {
            "answer": result["generation"],
            "sources": list(set(sources)) 
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ingest", summary="Ingest new documents")
async def trigger_ingestion():
    """Triggers the document chunking and vector embedding pipeline."""
    try:
        build_vector_store()
        return {"message": "Ingestion completed successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/documents", summary="List indexed documents")
async def list_documents():
    """Returns a list of unique document sources currently in ChromaDB."""
    try:
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
        
        
        doc_data = vectorstore.get(include=["metadatas"])
        
        sources = []
        for meta in doc_data.get("metadatas", []):
            if meta and "source" in meta:
                sources.append(meta["source"])
                
        return {"indexed_sources": list(set(sources))}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/feedback", summary="Submit feedback on an answer")
async def submit_feedback(request: FeedbackRequest):
    """Logs user feedback for a given answer."""
    
    print(f"\n--- FEEDBACK RECEIVED ---")
    print(f"Rating: {request.feedback}")
    print(f"Comment: {request.comment}")
    print(f"-------------------------\n")
    return {"message": "Feedback recorded successfully."}