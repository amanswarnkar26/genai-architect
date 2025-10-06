# agentic_rag_simplified.py
# Simple RAG workflow using LangGraph + Gemini

import os
from google.cloud import aiplatform
from langchain_google_vertexai import VertexAI, VertexAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langgraph.graph import StateGraph, END

# config
INDEX_NAME = "rag_kb_index"
PROJECT_ID = os.getenv("GCP_PROJECT_ID")
LOCATION = os.getenv("GCP_LOCATION", "us-central1")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

aiplatform.init(project=PROJECT_ID, location=LOCATION)
embed_model = VertexAIEmbeddings(model="models/gemini-embedding-001")

vectorstore = PineconeVectorStore(
    index_name=INDEX_NAME, embedding=embed_model, pinecone_api_key=PINECONE_API_KEY
)

llm = VertexAI(model="gemini-1.5-pro", temperature=0)

# nodes
def get_snippets(state):
    q = state["question"]
    docs = vectorstore.similarity_search(q, k=5)
    state["snippets"] = docs
    return state

def draft_answer(state):
    snips = "\n".join([f"[{d.metadata['doc_id']}] {d.page_content}" for d in state["snippets"]])
    prompt = f"Question: {state['question']}\nUse these notes:\n{snips}\nAnswer briefly with refs [KBxxx]."
    state["initial"] = llm.invoke(prompt)
    return state

def review_answer(state):
    prompt = f"Check this answer against the notes.\nAnswer: {state['initial']}\nSay COMPLETE or REFINE:<keywords>"
    state["review"] = llm.invoke(prompt)
    return state

def improve_answer(state):
    if state["review"].startswith("REFINE"):
        extra_q = state["review"].split("REFINE:")[-1].strip()
        extra = vectorstore.similarity_search(extra_q, k=1)
        all_snips = state["snippets"] + extra
        merged = "\n".join([f"[{d.metadata['doc_id']}] {d.page_content}" for d in all_snips])
        prompt = f"Refine answer using all:\n{merged}\nQ: {state['question']}\nFinal answer with refs."
        state["final"] = llm.invoke(prompt)
    else:
        state["final"] = state["initial"]
    return state

# graph
graph = StateGraph(dict)
graph.add_node("get_snippets", get_snippets)
graph.add_node("draft_answer", draft_answer)
graph.add_node("review_answer", review_answer)
graph.add_node("improve_answer", improve_answer)

graph.set_entry_point("get_snippets")
graph.add_edge("get_snippets", "draft_answer")
graph.add_edge("draft_answer", "review_answer")
graph.add_edge("review_answer", "improve_answer")
graph.add_edge("improve_answer", END)

rag = graph.compile()

if __name__ == "__main__":
    tests = [
        "What are best practices for caching?",
        "How should I set up CI/CD pipelines?",
        "What are performance tuning tips?",
        "How do I version my APIs?",
        "What should I consider for error handling?"
    ]
    for q in tests:
        result = rag.invoke({"question": q})
        print("\nQ:", q)
        print("Final Answer:", result["final"])
        print("Review:", result["review"])

