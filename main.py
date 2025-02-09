from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from smolagents import CodeAgent, DuckDuckGoSearchTool, LiteLLMModel
import os
import threading
lock = threading.Lock()  # Create a lock instance

app = FastAPI()

# Enable CORS - adjust allow_origins as needed for production.
app.add_middleware(
   CORSMiddleware,
   allow_origins=["*"],  # Allow all origins. Change for production.
   allow_credentials=True,
   allow_methods=["*"],
   allow_headers=["*"],
)

# Retrieve the Gemini API key from the environment variable
gemini_api_key = os.environ.get("GEMINI_API_KEY")
if gemini_api_key is None:
   raise EnvironmentError("GEMINI_API_KEY environment variable is not set!")

# Initialize the Gemini model and web search tool
model = LiteLLMModel(model_id="gemini/gemini-2.0-flash")
web_search_tool = DuckDuckGoSearchTool()
web_agent = CodeAgent(tools=[web_search_tool], model=model)

processing_lock = threading.Lock()  # ✅ Correct way

@app.get("/search")
def search(query: str = Query("search about smolagents web search agent what it can do", description="Search topic")):
   """
   This endpoint always:
     1. Uses a web search to gather the latest and most accurate information.
     2. Processes the search results using AI.
     3. Returns a clear, professional summary ending with confirmatory emojis.
   
   The use of a lock ensures multiple requests will be processed one after another.
   """
   task = f"""
   Query: {query}

   1. Web Search:
      - Regardless of the query type, first perform a web search using the provided tool.
      - Retrieve the most recent 5 sites, accurate, and relevant information from trusted sources (e.g., official websites, reputable news outlets, government data).
      - Extract and summarize the key details from these sources.

   2. AI Processing:
      - Analyze and process the gathered information to produce a concise and professional summary.
      - Explain any complex points in a clear and straightforward manner.

   3. Final Formatting:
      - Structure the output with short paragraphs or points for enhanced readability. If links are available or need to be mentioned, include them.
   """
   
   # Acquire the lock so that only one request is processed at a time.
   with processing_lock:
      result = web_agent.run(task)
   
   return {"result": result}
