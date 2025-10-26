# Loan Navigator Agent Suite (Capstone)

Multi-agent AI system for fintech loan support: SQL Analyst, Policy Guru (RAG), What-If Calculator, and Supervisor.
Built with FastAPI, deployable to **Google Cloud Run**, using **Vertex AI** (Gemini) in production.

## Repo Structure
```
.
├─ api/
│  └─ main.py
├─ agents/
│  ├─ calculator.py
│  ├─ policy_guru.py
│  ├─ sql_analyst.py
│  └─ supervisor.py
├─ core/
│  ├─ config.py
│  ├─ logging.py
│  └─ utils.py
├─ data/
│  ├─ loan.db            # from resources
│  ├─ policies/          # policy PDFs
│  
├─ requirements.txt
├─ Dockerfile
└─ README.md
```

## Local Run
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn api.main:app --host 0.0.0.0 --port 8080
# open http://localhost:8080/docs
```

### Example Request
```bash
curl -X POST http://localhost:8080/ask -H "Content-Type: application/json" -d '{
  "query": "What is my EMI status?",
  "params": {"customer_id": "CUST123"}
}'
```

## GCP Deployment (Cloud Run)
1. **Enable APIs**
   - Vertex AI, Cloud Run, Artifact Registry, Secret Manager, Cloud Logging.
2. **Auth**
   ```bash
   gcloud auth login
   gcloud config set project <PROJECT_ID>
   ```
3. **Build & Push**
   ```bash
   gcloud artifacts repositories create loan-nav --repository-format=docker --location=us --description="Loan Navigator images"
   gcloud builds submit --tag us-docker.pkg.dev/<PROJECT_ID>/loan-nav/loan-navigator:v1 .
   ```
4. **Deploy**
   ```bash
   gcloud run deploy loan-navigator --image us-docker.pkg.dev/<PROJECT_ID>/loan-nav/loan-navigator:v1      --region=us-central1 --allow-unauthenticated --port=8080
   ```
   (For private access, use IAP or IAM and remove `--allow-unauthenticated`.)

5. **Secrets** (recommended)
   - Store DB URI, API keys in Secret Manager and mount as env vars:
   ```bash
   gcloud secrets create DATABASE_URI --data-file=- <<< "sqlite:///data/loan.db"
   gcloud run services update loan-navigator --update-env-vars=DATABASE_URI=sm://DATABASE_URI
   ```

## Observability
- **Google Cloud Logging** enabled via `ENABLE_GOOGLE_LOGGING=true` env.
- **Langfuse**: set `LANGFUSE_HOST`, `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY` env vars to trace LLM spans.

## Testing
```bash
pytest -q
```

## Notes
- In production, replace heuristic NL→SQL with **Vertex AI** generated SQL guarded by **strict whitelist** and parameter binding.
- Replace placeholder policy RAG with real **Chroma** client loading `data/chroma` and returning scored citations.
