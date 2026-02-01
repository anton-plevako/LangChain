from dotenv import load_dotenv
from typing import List
from pydantic import BaseModel, Field
load_dotenv()

from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch

class Source(BaseModel):
    """Schema for a source used by the agent"""

    url:str = Field(description='The url of the source')

class AgentResponse(BaseModel):
    """Schema for the agent response"""

    answer:str = Field(description="The agent's answer to the query")
    sources:list[Source] = Field(default_factory=list, description='The list of sources used to generate the answer')

search = TavilySearch()

llm = ChatOpenAI()
tools = [search]
agent = create_agent(model=llm, tools=tools, response_format=AgentResponse)

def main():
    print("Hello from langchain-course!")
    result = agent.invoke({"messages": HumanMessage(content="search for 3 job postings using langchain in the bay area on linkdin and list their details in a summary")})
    print(result)

if __name__ == "__main__":
    main()
