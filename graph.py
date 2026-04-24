import os
import time
import os
from typing import List, TypedDict
from typing import List, TypedDict
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langgraph.graph import END, StateGraph, START

from langchain_community.tools.tavily_search import TavilySearchResults
load_dotenv()

from langchain_groq import ChatGroq

llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)


embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

class GraphState(TypedDict):
    """Represents the state of our graph."""
    question: str
    generation: str
    documents: List[Document]
    retries: int

def retrieve(state: GraphState):
    """Retrieves documents from the vector store."""
    print("---NODE: RETRIEVE---")
    question = state["question"]
    documents = retriever.invoke(question)
    
    return {"documents": documents, "question": question, "retries": state.get("retries", 0)}

def grade_documents(state: GraphState):
    """Determines whether the retrieved documents are relevant to the question."""
    print("---NODE: GRADE DOCUMENTS---")
    question = state["question"]
    documents = state["documents"]

    class Grade(BaseModel):
        binary_score: str = Field(description="Relevance score 'yes' or 'no'")

    structured_llm_grader = llm.with_structured_output(Grade)
    system = """You are a strict grader assessing relevance of a retrieved document to a user question. \n 
    If the document contains keywords or semantic meaning related to the question, grade it as 'yes'. \n
    Otherwise, grade it as 'no'."""
    
    grade_prompt = ChatPromptTemplate.from_messages([
        ("system", system),
        ("human", "Retrieved document: \n\n {document} \n\n User question: {question}"),
    ])
    
    retrieval_grader = grade_prompt | structured_llm_grader

    filtered_docs = []
    for d in documents:
        score = retrieval_grader.invoke({"question": question, "document": d.page_content})
        if score.binary_score == "yes":
            print("   -> Document Relevant")
            filtered_docs.append(d)
        else:
            print("   -> Document Irrelevant")

    return {"documents": filtered_docs, "question": question}

def rewrite_query(state: GraphState):
    """Rewrites the question to improve retrieval if previous attempts failed."""
    print("---NODE: REWRITE QUERY---")
    question = state["question"]
    retries = state.get("retries", 0)

    system = """You are a question re-writer optimized for vectorstore retrieval. 
    Look at the input question and reason about the underlying semantic intent. 
    Output an improved, clearer version of the question."""
    
    re_write_prompt = ChatPromptTemplate.from_messages([
        ("system", system),
        ("human", "Initial question: \n\n {question} \n Formulate an improved question."),
    ])
    
    question_rewriter = re_write_prompt | llm
    response = question_rewriter.invoke({"question": question})
    
    return {"question": response.content, "retries": retries + 1}
def web_search(state: GraphState):
    """
    Performs a web search based on the question if the vector database fails.
    """
    print("---NODE: WEB SEARCH---")
    question = state["question"]
    documents = state["documents"] 

    web_search_tool = TavilySearchResults(k=2)
    
    docs = web_search_tool.invoke({"query": question})

    web_content = "\n".join([d["content"] for d in docs])
    web_document = Document(page_content=web_content, metadata={"source": "Live Web Search (Tavily)"})
    
    documents.append(web_document)

    return {"documents": documents, "question": question}

def generate(state: GraphState):
    """Generates the final answer using the relevant documents."""
    print("---NODE: GENERATE---")
    question = state["question"]
    documents = state["documents"]

    system = """You are an AI assistant. Use the provided retrieved context to answer the question. 
    If you don't know the answer, just say that you don't know. Keep the answer concise.
    IMPORTANT: Include citations at the end of your answer by referencing the source URL provided in the context."""
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system),
        ("human", "Context: {context} \n\n Question: {question}")
    ])

    formatted_docs = "\n\n".join(
        f"Source: {doc.metadata.get('source', 'Unknown')}\nContent: {doc.page_content}" 
        for doc in documents
    )
    
    rag_chain = prompt | llm
    response = rag_chain.invoke({"context": formatted_docs, "question": question})
    
    return {"generation": response.content, "documents": documents, "question": question}

def decide_to_generate(state: GraphState):
    """Determines whether to generate an answer, re-write, or web search."""
    filtered_documents = state["documents"]
    retries = state.get("retries", 0)

    if not filtered_documents:
        if retries >= 2:
            print("---EDGE: RETRY LIMIT REACHED -> FALLBACK TO WEB SEARCH---")
            return "web_search"  # <--- WE CHANGED THIS
        else:
            print("---EDGE: NO RELEVANT DOCS -> REWRITE QUERY---")
            return "rewrite_query"
    else:
        print("---EDGE: DOCS RELEVANT -> GENERATE---")
        return "generate"


workflow = StateGraph(GraphState)

workflow.add_node("retrieve", retrieve)
workflow.add_node("grade_documents", grade_documents)
workflow.add_node("rewrite_query", rewrite_query)
workflow.add_node("generate", generate)
workflow.add_node("web_search", web_search)

workflow.add_edge(START, "retrieve")
workflow.add_edge("retrieve", "grade_documents")

workflow.add_conditional_edges(
    "grade_documents",
    decide_to_generate,
    {
        "rewrite_query": "rewrite_query",
        "web_search": "web_search",
        "generate": "generate",
    },
)

workflow.add_edge("rewrite_query", "retrieve")
workflow.add_edge("generate", END)
workflow.add_edge("web_search", "generate")

app = workflow.compile()


if __name__ == "__main__":
    
    print("\n\n=== TESTING RELEVANT QUERY ===")
    inputs = {"question": "How do I initialize a tensor in PyTorch?", "retries": 0}
    for output in app.stream(inputs):
        pass 
    
    print(f"\nFINAL ANSWER:\n{output['generate']['generation']}")

    print("\n\n=== TESTING IRRELEVANT QUERY ===")
    inputs = {"question": "What is the best recipe for chocolate chip cookies?", "retries": 0}
    for output in app.stream(inputs):
        pass 
    
    print(f"\nFINAL ANSWER:\n{output['generate']['generation']}")