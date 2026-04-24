# Express Analytics: Agentic RAG API

Name : Vineet Nagrale (IIT Kanpur)

---

This repository contains my submission for the **AI/ML Engineer Intern** take-home assignment at Express Analytics.It is a Retrieval-Augmented Generation (RAG) based Technical Documentation Assistant built using **FastAPI** and **LangGraph**.

The system goes beyond standard linear RAG by employing a self-corrective Agentic workflow.It ingests PyTorch documentation, evaluates its own retrieved context, rewrites queries if necessary, and includes two advanced fallback mechanisms: a Hallucination Checker and a Live Web Search via Tavily.

##  Key Features

- **Vector Database:** ChromaDB with HuggingFace `all-MiniLM-L6-v2` embeddings.
- **Agentic Workflow:** Built with LangGraph StateGraphs.
- **Self-Correction:** Evaluates retrieved documents.If irrelevant, it rewrites the query and tries again.
- **Bonus 1 - Hallucination Checker:** Audits the final LLM generation against the retrieved documents to prevent fabricated facts.
- **Bonus 2 - Web Search Fallback:** If local technical docs fail twice, it uses the Tavily API to search the live web and formats the results for generation.
- **REST API:** Fully featured backend with `/query`, `/ingest`, `/documents`, and `/feedback` endpoints.

---

### System Architecture

The LangGraph workflow consists of the following routing logic:
1.  **Retrieve:** Fetches top-k chunks from ChromaDB.
2. **Grade Documents:** An LLM strict-grades the chunks as "relevant" or "irrelevant".
3. **Routing (Conditional Edge):** 
    * If relevant $\rightarrow$ Proceed to Generate.
    * If irrelevant $\rightarrow$ Rewrite the query and loop back to Retrieve.
    * If max retries reached $\rightarrow$ Trigger Web Search fallback.
4.  **Generate:** Creates the answer using the filtered context.
5.  **Check Hallucinations:** Audits the answer. If the model hallucinates, it forces a rewrite.If grounded, it returns the final response.

---

###  Setup Instructions

**1. Clone the repository and navigate to the directory:**
```bash
git clone https://github.com/vineet1001/Vineet_Nagrale_Assignment_Express_Analytics.git
cd Vineet_Nagrale_Assignment_Express_Analytics
```
**2. Create and activate a virtual environment:**

```bash
python -m venv rag_env

# On Windows:
rag_env\Scripts\activate

# On Mac/Linux:
source rag_env/bin/activate
```
**3. Install dependencies:**

```bash
pip install -r requirements.txt
```
**4. Set up Environment Variables:**

```bash
GROQ_API_KEY=your_groq_key_here
TAVILY_API_KEY=your_tavily_key_here
```

### **How to Run**
1. Start the FastAPI server:
```bash
uvicorn main:app --reload
```

***2. Access the Interactive API Dashboard:***

Open your browser and navigate to http://127.0.0.1:8000/docs to use the Swagger UI.

**3. Ingest Documents:**

Before querying, trigger the POST /ingest endpoint to build the vector database from the provided PyTorch corpus.

**API Endpoints & Examples**
```bash
POST /query
Submits a question and returns an AI-generated answer with source citations.
```
```bash
Request:

JSON
{
  "question": "How do I build a simple neural network using PyTorch?"
}
```
```bash
Response:

JSON
{
  "answer": "To build a simple neural network in PyTorch, you need to define a class that inherits from torch.nn.Module...",
  "sources": ["pytorch_docs.pdf"]
}
```
```bash
POST /ingest
Triggers the chunking and embedding pipeline to populate ChromaDB.
```

```bash
GET /documents
Returns a list of unique document sources currently indexed in the vector store.
```
```bash
POST /feedback
Accepts user feedback for a given response.
```
```bash
Request:

JSON
{
  "feedback": "thumbs_up",
  "comment": "Great explanation!"
}
```
----

## How It Works: The Agentic RAG Flow

Here is a brief, step-by-step breakdown of how a user's question is processed from start to finish:

1. **The Request:** The user submits a question via the `POST /query` FastAPI endpoint.
2. **Retrieval:** The LangGraph workflow starts by searching the ChromaDB vector database for the most relevant technical documents.
3. **Document Grading:** Instead of blindly trusting the search, an LLM evaluates the retrieved chunks. Irrelevant documents are immediately discarded.
4. **Smart Routing:**
   * *Success:* If relevant documents are found, it routes to **Generation**.
   * *Retry:* If all documents are irrelevant, it rewrites the user's query and tries searching again.
   * *Fallback:* If it fails twice, it triggers a **Live Web Search (Tavily)** to pull the answer from the internet.
5. **Generation:** The LLM drafts a concise answer using only the verified context (either from the database or the web).
6. **Hallucination Check:** Before returning the answer, a final LLM audit ensures no fake facts were invented. If it detects a hallucination, it forces the system to start over.
7. **The Output:** The API delivers a clean JSON response containing the audited answer and its source citations.
---
##  Design Decisions & Tradeoffs

**1. Architecture & Workflow Reasoning**
* **Why LangGraph?** Traditional RAG pipelines using LangChain Expression Language (LCEL) are strictly linear—if the retrieval step fails to find good context, the generation step is guaranteed to fail or hallucinate. I chose LangGraph because it allows for a stateful, cyclic workflow. By treating the LLM as an agent, the system evaluates its own retrieved documents, loops back to rewrite the user's query if the documents are poor, and runs a final hallucination check before returning an answer.
* **LLM Provider (Groq / Llama 3.1):** Agentic RAG requires multiple rapid inference calls per query (e.g., query rewriting, document grading, generation, hallucination checking). I opted for Groq's API because it provides incredibly fast inference speeds and a generous free tier, preventing rate-limit crashes during the cyclic LangGraph loops that often occur with OpenAI or Google's standard free tiers.

**2. Chunking & Embedding Strategy**
* **Chunk Size & Overlap:** I utilized a chunk size of `1000` with an overlap of `200` using a Recursive Character Text Splitter. 
* **Reasoning:** Technical documentation (like PyTorch) relies heavily on code snippets, complex mathematical explanations, and nested bullet points. Smaller chunk sizes (e.g., 250-500) run the risk of severing a code block in half, stripping it of its surrounding explanation. A size of 1000 ensures the semantic meaning and context of technical implementations remain intact.
* **Embeddings:** I used HuggingFace's `all-MiniLM-L6-v2`. It is lightweight, runs locally without API costs, and is highly effective for semantic search across standard technical English.

**3. Assumptions Made**
* **Language & Scope:** I assumed that all user queries would be in English and directly related to the ingested PyTorch technical corpus.
* **Parsing Complexity:** I assumed that the provided PyTorch documentation was text-heavy enough that standard recursive character text splitting would be sufficient. For a larger, purely code-based repository, a specialized AST (Abstract Syntax Tree) code-parser would be necessary.
* **Database Scale:** I assumed a local, ephemeral ChromaDB instance was sufficient for a 3-5 document corpus. In a production environment, this would be swapped for a managed vector database like Pinecone or standard PostgreSQL with pgvector.

**4. Future Improvements (With More Time)**
* **Conversation Memory:** The current `/query` endpoint treats each request as a standalone interaction. I would implement session-based memory so the agent could handle follow-up questions (e.g., "Can you explain that last parameter in more detail?").
* **Frontend UI:** While the FastAPI Swagger dashboard is functional, I would build a minimal Streamlit or Gradio frontend to provide a cleaner, more intuitive chat interface for end-users.
* **Advanced Routing:** I would add a "Query Classifier" node at the very beginning to determine if a query is a "Greeting/Chitchat", "Direct API Reference", or "Conceptual Explanation", routing to different optimized prompts accordingly.
