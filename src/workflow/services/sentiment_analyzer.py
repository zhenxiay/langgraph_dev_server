"""This module provides a class with langchain framework to summarize large datasets with single API calls."""
import asyncio
import logging
from typing import List

import numpy as np
import pandas as pd
from langchain_core.output_parsers import StrOutputParser
from tqdm.auto import tqdm
from workflow.utils.llm import BaseLLMService
from workflow.utils.prompt_template import get_sentiment_analysis_prompt

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Disable httpx INFO logs
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)

class SentimentAnalyzer(BaseLLMService):
    """Analyzes sentiment of notes from Excel files using OpenAI and LangChain."""
    
    def __init__(
        self,
        provider: str = "AzureOpenAI",
        model: str = "gpt-4.1",
        temperature: float = 0
    ):
        """Initialize the sentiment analyzer class.
        
        Args:
            provider: LLM provider ("AzureOpenAI" or "OpenAI")
            model: Model name (e.g., "gpt-4o-mini", "gpt-4", "gpt-3.5-turbo")
            temperature: LLM temperature (0 = deterministic)
        """
        super().__init__(provider, model, temperature)

        # Create chains using RunnableSequence (pipe operator)
        self.sentiment_analysis_chain = get_sentiment_analysis_prompt() | self.llm | StrOutputParser()
        logger.info(f"Excel Notes Sentiment Analyzer initialized with provider-model: {provider}-{model}")

    def _create_input_list(
            self,
            df: pd.DataFrame,
            text_column: str='review',
            ) -> List[dict]:
        """
        Create input list for batch processing.

        Args:
            df (pd.DataFrame): DataFrame containing the text data
            text_column (str, optional): Column name containing the text. Defaults to 'review'.
        Returns:
            List[dict]: List of dictionaries for batch input        
        """
        text_list = [text for text in df[text_column].tolist()]

        batch_inputs = []

        for text in text_list:
            batch_inputs.append({"text": text})

        return batch_inputs

    async def async_analyze_sentiment(
            self,
            df: pd.DataFrame,
            text_column: str='review',
            ) -> pd.DataFrame:
        """
        Asynchronously summarize text data in a DataFrame.

        Args:
            df (pd.DataFrame): DataFrame containing the text data
            text_column (str, optional): Column name containing the text. Defaults to 'review'.

        Returns:
            pd.DataFrame: DataFrame containing the summarized texts
        """

        batch_inputs = self._create_input_list(df, text_column)
    
        batch_outputs = await self.sentiment_analysis_chain.abatch(batch_inputs, return_exceptions=True)

        return df.assign(Sentiment=batch_outputs)