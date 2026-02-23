import os
from dotenv import load_dotenv
from operator import itemgetter
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

load_dotenv()

print("Init components..")

embeddings = OpenAIEmbeddings()
llm = ChatOpenAI()

vector_store = PineconeVectorStore(
    index_name=os.environ['INDEX_NAME'], embedding=embeddings)
retriever = vector_store.as_retriever(search_kwargs={"k":3})

prompt_template = ChatPromptTemplate.from_template(
    """answer the question based only on the following context:
    
    {context}
    
    Question: {query}
    
    Provide a detailed answer:"""
)

def format_docs(docs):
    """format retrieved documents into a single string."""
    return "\n\n".join(doc.page_content for doc in docs)

def retrieval_chain_without_lcel(query: str):
    """
    --
    """
    docs = retriever.invoke(query)
    context = format_docs(docs)

    messages = prompt_template.format_messages(context=context,
                                               query=query)
    response = llm.invoke(messages)
    return response.content


def create_retrieval_with_lcel():
    retreival_chain = (
        RunnablePassthrough.assign(
            context=itemgetter("query") | retriever | format_docs
         ) 
        | prompt_template
        | llm
        | StrOutputParser()
    )
    return retreival_chain

if __name__ == "__main__":
    print("Retrieving...")

    query = "what is Pinecone in machine learning?"

    # print(llm.invoke([HumanMessage(content=query)]).content)

    # print(retrieval_chain_without_lcel(query=query))

    query_dict = {"query": "what is Pinecone?"}
    chain_with_lcel = create_retrieval_with_lcel()
    result_with_lcel = chain_with_lcel.invoke(query_dict)
    print(result_with_lcel)

    

