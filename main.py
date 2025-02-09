from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from smolagents import CodeAgent, DuckDuckGoSearchTool, LiteLLMModel
import os

app = FastAPI()

# Enable CORS - you can adjust allow_origins to restrict to specific domains
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins. Change this for production.
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Get the Gemini API key from the environment variable
gemini_api_key = os.environ.get("GEMINI_API_KEY")
if gemini_api_key is None:
    raise EnvironmentError("GEMINI_API_KEY environment variable is not set!")

# Initialize the Gemini model and tools
model = LiteLLMModel(model_id="gemini/gemini-2.0-flash")
web_search_tool = DuckDuckGoSearchTool()
web_agent = CodeAgent(tools=[web_search_tool], model=model)

@app.get("/search")
def search(query: str = Query("search about smolagents web search agent what it can do", description="Search topic")):
    """
    This endpoint searches the web and returns a professional summary.
    """
    task = f"""
    Query: {query}

    1. Identify the Query Type:
       - If the query requires real-time or updated data (e.g., currency conversion, stock prices, latest news, weather), use a web search.
       - If the query is factual knowledge (e.g., science facts, historical events, definitions), answer directly if known.
       - If the query involves complex reasoning, analysis, or calculations, break it down step by step before answering.

    2. For Web-Based Queries:
       - Search the web and extract the most relevant, up-to-date, and accurate information.
       - Prioritize trusted sources (official websites, reputable news platforms, government data, etc.).
       - Summarize findings in a concise and clear format.

    3. For Fact-Based Queries:
       - If it is a well-known fact, answer directly and accurately.
       - If fact-checking is needed, compare multiple sources before responding.
       - Provide clarity and explanations when necessary.

    4. For Analytical or Computational Queries:
       - Break down the problem logically.
       - Perform necessary calculations if required.
       - Explain the reasoning in simple steps.

    5. Format the Output for Readability:
       - Use short paragraphs, clear points, and structured responses.
       - Add line breaks and emojis for better readability where appropriate.
       - Keep responses precise and professional yet easy to understand.
    """
    result = web_agent.run(task)
    return {"result": result}