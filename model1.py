import os

from dotenv import load_dotenv
from openai import AzureOpenAI
from langchain_openai import AzureChatOpenAI

load_dotenv()


def initialize_llm(llm_choice):
    if llm_choice == "gpt4o":
        try:
            # secrets.toml 형태의 전체 URL을 사용하는 경우
            endpoint_url = os.getenv("AZURE_OPENAI_ENDPOINT")
            if endpoint_url and "chat/completions" in endpoint_url:
                # URL에서 base endpoint 추출
                base_endpoint = endpoint_url.split("/openai/deployments")[0]
                return AzureOpenAI(
                    azure_endpoint=base_endpoint,
                    azure_deployment=os.getenv("GPT4O_DEPLOYMENT"),
                    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
                    api_version="2025-01-01-preview",
                    temperature=0,
                )
        except Exception:
            pass

        # 기존 환경변수 형태를 사용하는 경우 (fallback)
        return AzureOpenAI(
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            azure_deployment=os.getenv("GPT4O_DEPLOYMENT"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version="2024-12-01-preview",
            temperature=0,
        )

    if llm_choice == "langchain_gpt4o":
        try:
            # secrets.toml 형태의 전체 URL을 사용하는 경우
            endpoint_url = os.getenv("AZURE_OPENAI_ENDPOINT")
            if endpoint_url and "chat/completions" in endpoint_url:
                # URL에서 base endpoint 추출
                base_endpoint = endpoint_url.split("/openai/deployments")[0]
                return AzureChatOpenAI(
                    azure_endpoint=base_endpoint,
                    azure_deployment=os.getenv("GPT4O_DEPLOYMENT"),
                    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
                    api_version="2025-01-01-preview",
                    streaming=True,
                    temperature=0,
                )
        except Exception:
            pass

        # 기존 환경변수 형태를 사용하는 경우 (fallback)
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
