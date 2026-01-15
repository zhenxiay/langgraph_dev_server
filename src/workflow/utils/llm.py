"""
This module define the base model which will be used by the workflow automation.
"""
import polars as pl
from typing import List
import logging
from langchain_openai import AzureChatOpenAI, ChatOpenAI
from langchain_core.runnables import RunnableConfig, RunnableSequence

from workflow.utils.timer import log_time

def _get_openai_llm(
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

def _get_azure_openai_llm(
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

class BaseModel:
    """
    Base class for services using LLM providers.
    """
    
    @log_time
    def __init__(
        self,
        provider: str = "AzureOpenAI",
        model: str = "gpt-4.1",
        temperature: float = 0,
        max_retries: int = 2,
        max_concurrent_requests: int = 128
    ):
        """Initialize the LLM service.
        
        Args:
            provider: LLM provider ("AzureOpenAI" or "OpenAI")
            model: Model name (e.g., "gpt-4o-mini", "gpt-4", "gpt-3.5-turbo")
            temperature: LLM temperature (0 = deterministic)
            max_concurrent_requests: Maximum number of concurrent requests for batch processing
        """
        if provider == "AzureOpenAI":
            self.llm = _get_azure_openai_llm(
                model=model,
                temperature=temperature,
                max_retries=max_retries
            )
        else:
            self.llm = _get_openai_llm(
                model=model,
                temperature=temperature,
                max_retries=max_retries
            )
        
        self.provider = provider
        self.model = model
        self.max_concurrent_requests = max_concurrent_requests

        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    @log_time
    def _create_input_list(
            self,
            df: pl.DataFrame,
            text_column: str='review',
            ) -> List[dict]:
        """
        Create input list for batch processing.

        Args:
            df (pl.DataFrame): Polars DataFrame containing the text data
            text_column (str, optional): Column name containing the text. Defaults to 'review'.
        Returns:
            List[dict]: List of dictionaries for batch input        
        """
        text_list = df.select(text_column).to_series().to_list()

        batch_inputs = []

        for text in text_list:
            batch_inputs.append({"text": text})

        return batch_inputs
    
    @log_time
    def _extract_text(
            self,
            df: pl.DataFrame,
            text_column: str='review',
            length: int=500,
            option: str='summary'
            ) -> pl.DataFrame:
        """
        Extract summary texts from DataFrame based on given length limit.

        Retruns a Polars DataFrame filtered by text length.
        """

        df= df.with_columns(
            pl.col(text_column).str.len_chars().alias("text_length")
        )

        if option == 'summary':
            return df.filter(pl.col("text_length") > length)
        elif option == 'translation':
            return df.filter(pl.col("text_length") <= length)

    @log_time
    async def _async_process_current_batch(
            self,
            current_batch: List[dict],
            llm_chain: RunnableSequence = None,
            ) -> List[str]:
        """
        Asynchronously process the current batch with a given llm chain.

        Args:
            current_batch (List[dict]): List of dictionaries for current batch input

        Returns:
            List[str]: List of outputs from the LLM
        """

        response = await llm_chain.abatch(
                                current_batch, 
                                return_exceptions=True, 
                                config=RunnableConfig(max_concurrency=self.max_concurrent_requests)
                                )
        
        return response
        
    async def _async_batch_llm_request(
            self,
            batch_inputs: List[dict],
            BATCH_SIZE: int = 32,
            llm_chain: RunnableSequence = None,
            ) -> List[str]:
        """
        Asynchronously process batch requests with a given llm chain.

        Args:
            batch_inputs (List[dict]): List of dictionaries for batch input
            BATCH_SIZE (int, optional): Number of samples to process in each batch. Defaults to 32.

        Returns:
            List[str]: List of outputs from the LLM
        """

        batch_outputs = []
        
        for i in range(0, len(batch_inputs), BATCH_SIZE):
            current_batch = batch_inputs[i:i + BATCH_SIZE]
            
            response = await self._async_process_current_batch(
                current_batch=current_batch,
                llm_chain=llm_chain
            )
            
            batch_outputs.extend(response)

            self.logger.info(f"Processed batch {i // BATCH_SIZE + 1} / {((len(batch_inputs) - 1) // BATCH_SIZE) + 1}")
        
        return batch_outputs