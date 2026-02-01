from dotenv import load_dotenv

from prompt import REACT_PROMPT_WITH_FORMAT_INSTRUCTIONS
from schemas import AgentResponse

load_dotenv()

from langchain_classic import hub
from langchain_classic.agents import AgentExecutor
from langchain_classic.agents.react.agent import create_react_agent
from langchain_core.output_parsers.pydantic import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch

tools = [TavilySearch()]
llm = ChatOpenAI(model="gpt-4")
# react_prompt = hub.pull("hwchase17/react")
output_parser = PydanticOutputParser(pydantic_object=AgentResponse)
react_prompt_with_format_instructions = PromptTemplate(
    template=REACT_PROMPT_WITH_FORMAT_INSTRUCTIONS,
    input_variables=["input", "agent_scratchpad", "tools", "tool_names"],
).partial(format_instructions=output_parser.get_format_instructions())
agent = create_react_agent(
    llm=llm, tools=tools, prompt=react_prompt_with_format_instructions
)
agent_executor = AgentExecutor(
    agent=agent, tools=tools, verbose=True, handle_parsing_errors=True
)

extract_output = RunnableLambda(lambda x: x["output"])
parse_output = RunnableLambda(lambda x: output_parser.parse(x))
chain = agent_executor | extract_output | parse_output


def main():
    print("Hello from langchain-course!")
    result = chain.invoke(
        input={
            "input": "search for 3 job postinig for an ai engineer with langchain as one of the skillsets in the bay area on linkdin and list a short summary of their details."
        }
    )
    print(result)
    print("done")


if __name__ == "__main__":
    main()
