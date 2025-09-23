import os
import mlflow
from langchain.chains import SequentialChain
from langchain.prompts import PromptTemplate
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain.llms import VertexAI
from langchain.tools.yahoo_finance_news import YahooFinanceNewsTool
from langchain.tools.yahoo_finance_ticker import YahooFinanceTickerTool

# --------------------------
# MLflow setup
# --------------------------
mlflow.set_tracking_uri("http://localhost:5000")   # change if remote
mlflow.set_experiment("market_sentiment_chain")

# --------------------------
# Step 1 & 2: Stock Ticker Lookup
# --------------------------
def get_stock_code(company_name: str):
    with mlflow.start_run(nested=True, run_name="stock_lookup"):
        tool = YahooFinanceTickerTool()
        result = tool.run(company_name)
        mlflow.log_param("company_input", company_name)
        mlflow.log_metric("tickers_found", len(result))
        return result[0] if result else None

# --------------------------
# Step 3: Fetch News
# --------------------------
def get_company_news(stock_code: str):
    with mlflow.start_run(nested=True, run_name="news_fetch"):
        tool = YahooFinanceNewsTool()
        news = tool.run(stock_code)
        mlflow.log_param("stock_code", stock_code)
        mlflow.log_metric("articles_fetched", len(news))
        return news

# --------------------------
# Step 4: Sentiment Analysis via Gemini
# --------------------------
def analyze_sentiment(company_name: str, stock_code: str, news: list):
    response_schemas = [
        ResponseSchema(name="company_name", description="Company full name"),
        ResponseSchema(name="stock_code", description="Ticker code"),
        ResponseSchema(name="newsdesc", description="Concise description of news"),
        ResponseSchema(name="sentiment", description="Positive/Negative/Neutral"),
        ResponseSchema(name="people_names", description="List of people mentioned"),
        ResponseSchema(name="places_names", description="List of places mentioned"),
        ResponseSchema(name="other_companies_referred", description="Other companies"),
        ResponseSchema(name="related_industries", description="Industries referenced"),
        ResponseSchema(name="market_implications", description="Potential impact"),
        ResponseSchema(name="confidence_score", description="Confidence score 0.0-1.0")
    ]
    parser = StructuredOutputParser.from_response_schemas(response_schemas)
    format_instructions = parser.get_format_instructions()

    prompt = PromptTemplate(
        template=(
            "You are a financial analyst AI. Given the following recent news for {company_name} ({stock_code}), "
            "generate a structured JSON sentiment profile.\n\n"
            "News:\n{news}\n\n"
            "{format_instructions}"
        ),
        input_variables=["company_name", "stock_code", "news"],
        partial_variables={"format_instructions": format_instructions},
    )

    llm = VertexAI(model="gemini-2.0-flash", temperature=0.2)
    chain = prompt | llm | parser

    with mlflow.start_run(nested=True, run_name="sentiment_analysis"):
        result = chain.invoke({
            "company_name": company_name,
            "stock_code": stock_code,
            "news": news
        })
        mlflow.log_param("company_name", company_name)
        mlflow.log_param("stock_code", stock_code)
        mlflow.log_text(str(result), "sentiment_output.json")
        return result

# --------------------------
# Step 5: Main Function
# --------------------------
def run_pipeline(company_name: str):
    with mlflow.start_run(run_name=f"sentiment_chain_{company_name}"):
        stock_code = get_stock_code(company_name)
        if not stock_code:
            raise ValueError("No stock ticker found for company")

        news = get_company_news(stock_code)
        sentiment = analyze_sentiment(company_name, stock_code, news)
        return sentiment

# --------------------------
# Entry Point
# --------------------------
if __name__ == "__main__":
    company = "Google"
    result = run_pipeline(company)
    print(result)
