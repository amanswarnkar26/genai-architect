# index_kb.py
import json
import os
from google.cloud import aiplatform
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain_google_vertexai import VertexAIEmbeddings

# ========== CONFIG ==========
INDEX_NAME = "agenticrag-kb"
PROJECT_ID = os.getenv("GCP_PROJECT_ID")
LOCATION = os.getenv("GCP_LOCATION", "us-central1")

# ===== Init Vertex AI =====
aiplatform.init(project=PROJECT_ID, location=LOCATION)

embeddings = VertexAIEmbeddings(model="models/gemini-embedding-001")

# ===== Load KB =====
with open("self_critique_loop_dataset.json", "r") as f:
    kb_data = json.load(f)

texts = [entry["answer_snippet"] for entry in kb_data]
metadatas = [{"doc_id": entry["doc_id"], "source": entry["source"]} for entry in kb_data]
ids = [entry["doc_id"] for entry in kb_data]

# ===== Init Pinecone =====
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
if not PINECONE_API_KEY:
    raise RuntimeError("Missing Pinecone API key.")

pc = Pinecone(api_key=PINECONE_API_KEY)

if INDEX_NAME not in [idx["name"] for idx in pc.list_indexes()]:
    dim = len(embeddings.embed_query("dimension probe"))
    pc.create_index(
        name=INDEX_NAME,
        dimension=dim,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )

vectorstore = PineconeVectorStore(
    index_name=INDEX_NAME,
    embedding=embeddings,
    pinecone_api_key=PINECONE_API_KEY,
)

vectorstore.add_texts(texts=texts, metadatas=metadatas, ids=ids)
print("KB indexed into Pinecone with Gemini embeddings.")
