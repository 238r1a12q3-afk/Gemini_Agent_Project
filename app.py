from langchain.agents import create_agent
from langchain.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import requests
import os


load_dotenv()


llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    temperature=0
)


@tool
def get_weather(city: str) -> str:
    """
    Fetch weather information of a city.
    """

    url = f"https://wttr.in/{city}?format=%C+%t"

    response = requests.get(url)

    if response.status_code == 200:
        return f"Weather in {city}: {response.text}"

    return "Unable to fetch weather"


agent = create_agent(
    model=llm,
    tools=[get_weather]
)


while True:

    user_query = input("\nAsk me anything: ")

    if user_query.lower() == "exit":
        break

    result = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": user_query
                }
            ]
        }
    )

    print(
        "\nAI:",
        result["messages"][-1].content
    )