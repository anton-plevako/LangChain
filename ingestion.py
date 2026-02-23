import os
import re
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter


load_dotenv()


def clean_transcript(text):
    text = re.sub(r"^\s*\d+\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"Heritage Reporting Corporation.*?Review", "", text)
    text = re.sub(r"\f", "", text)
    text = re.sub(r"\n(?![A-Z .,'-]+:)", " ", text)
    return text


def extract_speakers(text):
    matches = re.findall(r"\b([A-Z .,'-]+):", text)
    speakers = list(set([m.strip() for m in matches]))
    return speakers if speakers else ["UNKNOWN"]


if __name__ == '__main__':
    print("Ingesting...")

    loader = TextLoader(
        r"C:\Users\Anton\Projects\langchain-course\24-1287_b07d.txt",
        encoding='UTF-8'
    )

    document = loader.load()

    document[0].page_content = clean_transcript(document[0].page_content)

    print("splitting...")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=250,
        separators=[
            r"\n[A-Z .,'-]+:",
            "\n\n",
            "\n",
            ". ",
            " "
        ],
        is_separator_regex=True
    )

    texts = text_splitter.split_documents(document)

    for doc in texts:
        doc.metadata["speakers"] = extract_speakers(doc.page_content)

    print(f"created {len(texts)} chunks")

    embeddings = OpenAIEmbeddings(
        openai_api_key=os.environ.get('OPENAI_API_KEY')
    )

    print("Ingesting into Pinecone...")

    PineconeVectorStore.from_documents(
        texts,
        embeddings,
        index_name=os.environ['INDEX_NAME']
    )

    print("pinecone stored")