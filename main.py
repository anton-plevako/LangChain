from dotenv import load_dotenv

load_dotenv()

from langchain_classic import hub

from langchain_classic.agents import AgentExecutor
from langchain_classic.agents.react.agent import create_react_agent
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch

tools = [TavilySearch()]
llm = ChatOpenAI(model='gpt-4')
react_prompt = hub.pull("hwchase17/react")
agent = create_react_agent(
    llm=llm,
    tools=tools,
    prompt=react_prompt
)
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True)

chain = agent_executor

def main():
    print("Hello from langchain-course!")
    result = chain.invoke(
        input={
            "input": "search for 3 job postinig for an ai engineer with langchain as one of the skillsets in the bay area on linkdin and list a short summary of their details. Also make sure that it is open for hire. Preferably posted in the last week."
        }
    )
    print(result)
    print('done')

if __name__ == "__main__":
    main()
