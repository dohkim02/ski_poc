import os
import streamlit as st

from dotenv import load_dotenv
from openai import AzureOpenAI
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from azure.identity import DefaultAzureCredential

load_dotenv()


def initialize_llm(llm_choice):
    if llm_choice == "gpt4o":
        return AzureOpenAI(
            azure_endpoint=st.secrets["AZURE_OPENAI_ENDPOINT"],
            azure_deployment=st.secrets["GPT4O_DEPLOYMENT"],
            api_key=st.secrets["AZURE_OPENAI_API_KEY"],
            api_version="2024-12-01-preview",
            temperature=0,
        )
    if llm_choice == "langchain_gpt4o":
        try:
            return AzureChatOpenAI(
                azure_endpoint=st.secrets["AZURE_OPENAI_ENDPOINT"],
                azure_deployment=st.secrets["GPT4O_DEPLOYMENT"],
                api_key=st.secrets["AZURE_OPENAI_API_KEY"],
                api_version="2024-08-01-preview",
                streaming=True,
                temperature=0,
            )
        except:
            return AzureChatOpenAI(
                azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
                azure_deployment=os.getenv("GPT4O_DEPLOYMENT"),
                api_key=os.getenv("AZURE_OPENAI_API_KEY"),
                api_version="2024-08-01-preview",
                streaming=True,
                temperature=0,
            )

    else:
        raise ValueError(f"Unsupported LLM choice: {llm_choice}")


def initialize_embedding(embedding_choice):
    if embedding_choice == "text-embedding-3-large":
        return AzureOpenAIEmbeddings(
            azure_endpoint=os.getenv("EMBEDDING_003_ENDPOINT"),
            azure_deployment=os.getenv("EMBEDDING_003_LARGE"),
            api_key=os.getenv("EMBEDDING_003_KEY"),
            api_version=os.getenv("EMBEDDING_003_API_VERSION"),  # 추가
        )
    else:
        return None
