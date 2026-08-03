import certifi
import requests
import streamlit as st
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langchain.agents import (
    create_react_agent,
    AgentExecutor
)
from langchain import hub

from langchain_community.tools.tavily_search import TavilySearchResults

## Certifi is used to ensure that the SSL certificate verification works correctly when making HTTPS requests. This is particularly important when using APIs that require secure connections.
os.environ["SSL_CERT_FILE"] = certifi.where()
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
WEATHERSTACK_API_KEY = os.getenv("WEATHERSTACK_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

# streamlit page configuration
st.set_page_config(
    page_title="Agentic AI Assistant",
    page_icon="🤖",
    layout="centered"
)

st.title("🤖 Agentic AI Assistant")
st.markdown("Search + Weather AI Agent using LangChain")

# Define the tools for the agent

search_tool = TavilySearchResults(max_results=2)

# Define the weather tool using the @tool decorator

@tool
def get_weather_data(city: str) -> str:
    """
    Fetch current weather information for a city.
    """

    url = (
        f"http://api.weatherstack.com/current?"
        f"access_key={WEATHERSTACK_API_KEY}&query={city}"
    )

    response = requests.get(url)

    data = response.json()

    if "current" not in data:
        return f"Could not fetch weather data for {city}"

    return (
        f"City: {city}\n"
        f"Temperature: {data['current']['temperature']}°C\n"
        f"Weather: {data['current']['weather_descriptions'][0]}\n"
        f"Humidity: {data['current']['humidity']}%"
    )

#LLM configuration

llm = ChatOpenAI(
    model="openai/gpt-4o-mini",
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    temperature=0,
    max_tokens=1000
)

# Prompt configuration for the agent using LangChain Hub
prompt = hub.pull("hwchase17/react")

# Create the agent using the React framework
tools = [
    search_tool,
    get_weather_data
]

# Create the agent using the React framework
agent = create_react_agent(
    llm=llm,
    tools=tools,
    prompt=prompt
)

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True
)


user_query = st.text_input(
    "Enter your query:",
    placeholder="Example: Find the capital of India and current weather"
)

if st.button("Run Agent"):

    if user_query:

        with st.spinner("Agent is thinking..."):

            try:
                response = agent_executor.invoke({
                    "input": user_query
                })

                st.success("Response Generated")

                st.markdown("## Final Response")
                st.write(response["output"])

            except Exception as e:
                st.error(f"Error: {str(e)}")

    else:
        st.warning("Please enter a query")