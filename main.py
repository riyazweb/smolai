from fastapi import FastAPI, Query
from smolagents import CodeAgent, DuckDuckGoSearchTool, LiteLLMModel
import os

# Initialize FastAPI app
app = FastAPI()

# Set your Gemini API key (replace with your actual key)
os.environ["GEMINI_API_KEY"] = "AIzaSyDP4wC7FsIuMhYXpU_sPai-ry4zDW6N_tA"
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
    1. Use a web search to find the most recent and relevant articles on "{query}".
    2. Identify the top 5 articles that provide insights on the topic.
    3. For each article, extract the title, publication date, and a brief summary of the key insights.
    4. Combine all this information into one cohesive, professionally written summary. The summary should integrate all the details (titles, publication dates, and key insights) into a full-text narrative without listing them separately.
    5. The final output should be a single well-organized paragraph that comprehensively summarizes all the findings in short points with emojis and line breaks.
    """
    result = web_agent.run(task)
    return {"result": result}
