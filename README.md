# Real-Time Market Sentiment Analyzer Using LangChain

This project implements a LangChain-powered pipeline to analyze real-time market sentiment
for any given company. It integrates ticker lookup, news fetching, Gemini LLM sentiment analysis,
and mlflow-based observability.

---

## Features
- Accepts company name as input.
- Extracts/generates stock ticker symbol (Yahoo Finance).
- Fetches recent news articles for the ticker.
- Sends news summaries to **Google Gemini-2.0-flash** via LangChain.
- Produces a structured sentiment JSON.
- Logs all prompts, outputs, and metrics with **mlflow**.

---

## 🛠️ Tech Stack
- **Framework:** LangChain
- **LLM:** Google Gemini-2.0-flash (via Vertex AI integration)
- **Data Source:** Yahoo Finance tools / Brave Search (pluggable)
- **Prompt Management & Observability:** mlflow
- **Environment:** Python 3.10+

---

## ⚙️ Setup Instructions

### 1. Clone Repo
```bash
git clone https://github.com/yourname/market-sentiment-analyzer.git
cd market-sentiment-analyzer

2. Install Dependencies
pip install -r requirements.txt


requirements.txt

langchain
google-cloud-aiplatform
mlflow

3. Configure Gemini

Enable Vertex AI API in GCP.

Set your project and credentials:

export GOOGLE_CLOUD_PROJECT="your-project-id"
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/service_account.json"

4. Configure mlflow

Run mlflow server locally:

mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns

5. Run the Chain
python sentiment_chain.py
