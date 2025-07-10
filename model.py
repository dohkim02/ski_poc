import os
import streamlit as st

# from dotenv import load_dotenv
from openai import AzureOpenAI
from langchain_openai import AzureChatOpenAI

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
        return AzureChatOpenAI(
            azure_endpoint=st.secrets["AZURE_OPENAI_ENDPOINT"],
            azure_deployment=st.secrets["GPT4O_DEPLOYMENT"],
            api_key=st.secrets["AZURE_OPENAI_API_KEY"],
            api_version="2024-08-01-preview",
            streaming=True,
            temperature=0,
        )
    else:
        raise ValueError(f"Unsupported LLM choice: {llm_choice}")
