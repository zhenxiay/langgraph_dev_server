"""This module provides a class with langchain framework to summarize large datasets with single API calls."""
import asyncio
import logging
from typing import List

import numpy as np
import pandas as pd
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableConfig
from tqdm.auto import tqdm
from workflow.utils.llm import BaseModel
from workflow.utils.prompt_template import get_sentiment_analysis_prompt

# Disable httpx INFO logs
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)

class SentimentAnalyzer(BaseModel):
    """Analyzes sentiment of notes from Excel files using OpenAI and LangChain."""
    
    def __init__(
        self,
        provider: str = "AzureOpenAI",
        model: str = "gpt-4.1",
        temperature: float = 0,
        max_retries: int = 2,
        max_concurrent_requests: int = 256
    ):
        """Initialize the sentiment analyzer class.
        
        Args:
            provider: LLM provider ("AzureOpenAI" or "OpenAI")
            model: Model name (e.g., "gpt-4o-mini", "gpt-4", "gpt-3.5-turbo")
            temperature: LLM temperature (0 = deterministic)
            max_concurrent_requests: Maximum number of concurrent requests for batch processing
        """
        super().__init__(provider, model, temperature, max_retries, max_concurrent_requests)

        # Create chains using RunnableSequence (pipe operator)
        self.sentiment_analysis_chain = get_sentiment_analysis_prompt() | self.llm | StrOutputParser()
        self.logger.info(f"Excel Notes Sentiment Analyzer initialized with provider-model: {provider}-{model}")

    async def async_analyze_sentiment(
            self,
            df: pd.DataFrame,
            text_column: str='review',
            BATCH_SIZE: int = 32,
            ) -> pd.DataFrame:
        """
        Asynchronously summarize text data in a DataFrame.

        Args:
            df (pd.DataFrame): DataFrame containing the text data
            text_column (str, optional): Column name containing the text. Defaults to 'review'.
            BATCH_SIZE (int, optional): Number of samples to process in each batch. Defaults to 32.

        Returns:
            pd.DataFrame: DataFrame containing the summarized texts
        """

        batch_inputs = self._create_input_list(df, text_column)

        batch_outputs = await self._async_batch_llm_request(
            batch_inputs = batch_inputs, 
            BATCH_SIZE = BATCH_SIZE, 
            llm_chain = self.sentiment_analysis_chain
            )

        return df.assign(Sentiment=batch_outputs)