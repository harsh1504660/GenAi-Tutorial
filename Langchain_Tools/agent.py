

import os
import requests
import webbrowser
from playsound3 import playsound
from langchain.agents import Tool, initialize_agent, AgentType
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
# Set your OpenRouter API key

# Use OpenRouter with mistral
llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation",
)
model = ChatHuggingFace(llm=llm)
# Search Tool
from langchain_community.tools.tavily_search import TavilySearchResults

search_tool =TavilySearchResults(
    tavily_api_key='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'
    )


# Weather Tool

def get_weather_data(city: str) -> str:
    url = f"https://api.weatherstack.com/current?###access_key=###&query={city}"
    res = requests.get(url).json()
    return str(res)
def play_music(song_name: str) -> str:
    """
    Plays a local MP3 file given the filename (must include .mp3 extension).
    """
    playsound(r'Langchain_Tools\song.mp3')



song_tool = Tool(
    name="play_a_song",
    func=play_music,
    description="Play the song, the input should be the song name"
)
weather_tool = Tool(
    name="get_weather_data",
    func=get_weather_data,
    description="Get current weather for a city. Input should be a city name."
)

# Create agent
agent_executor = initialize_agent(
    tools=[search_tool, weather_tool,song_tool],
    llm=model,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
)

# Run agent
query = "Play the song Believer by Imagine Dragons"
response = agent_executor.run(query)
print("Agent Response:", response)
