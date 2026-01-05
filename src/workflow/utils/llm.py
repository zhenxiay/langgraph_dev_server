"""This module define large language models (LLMs) which will be used by the workflow automation."""
from langchain_openai import AzureChatOpenAI, ChatOpenAI

def get_openai_llm(
        model: str = "gpt-4.1",
        temperature: float = 0,
        max_retries: int = 2,
    ):
        """Initialize large language models (LLMs) which will be used by the workflow automation.
        
        Args:
            model: OpenAI model name
            temperature: LLM temperature (0 = deterministic)
        """
        # Use regular ChatOpenAI which supports ainvoke
        return ChatOpenAI(
            model_name=model,
            temperature=temperature,
            max_retries=max_retries
        )

def get_azure_openai_llm(
        model: str = "gpt-4.1",
        api_version: str = "2025-01-01-preview",
        azure_endpoint: str = "https://your-endpoint.openai.azure.com/",
        temperature: float = 0,
        max_retries: int = 2,
    ):
        """Initialize large language models (LLMs) which will be used by the workflow automation.
        
        Args:
            model: Azure OpenAI deployment name
            api_version: API version
            azure_endpoint: Azure OpenAI endpoint
            temperature: LLM temperature (0 = deterministic)
        """
        # Use regular AzureChatOpenAI which supports ainvoke
        import os
        azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", azure_endpoint)

        return AzureChatOpenAI(
            azure_deployment=model,
            api_version=api_version,
            azure_endpoint=azure_endpoint,
            temperature=temperature,
            max_retries=max_retries
        )