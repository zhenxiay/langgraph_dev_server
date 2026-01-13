"""This module provides a class with langchain framework to summarize large datasets with single API calls."""
import asyncio
import logging
from typing import List

import numpy as np
import pandas as pd
import polars as pl
from langchain_core.runnables import RunnableConfig
from langchain_core.output_parsers import StrOutputParser
from workflow.utils.llm import BaseModel
from workflow.utils.prompt_template import (
    get_summarization_prompt, 
    get_translation_prompt,
    get_full_translation_prompt,
)

# Disable httpx INFO logs
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)

class NotesSummarizer(BaseModel):
    """Summarizes notes from Excel files using OpenAI and LangChain."""
    
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
        self.summarization_chain = get_summarization_prompt() | self.llm | StrOutputParser()
        self.logger.info(f"Excel Notes Summarizer initialized with provider-model: {provider}-{model}")

        self.translation_chain = get_translation_prompt() | self.llm | StrOutputParser()
        self.logger.info(f"Excel Notes Translator initialized with provider-model: {provider}-{model}")

        self.full_translation_chain = get_full_translation_prompt() | self.llm | StrOutputParser()
        self.logger.info(f"Excel Notes Full Translator initialized with provider-model: {provider}-{model}")
        
    async def async_summarize_text(
            self,
            df: pl.DataFrame,
            text_column: str='text',
            length: int=500,
            BATCH_SIZE: int = 32,
            ) -> pl.DataFrame:
        """
        Asynchronously summarize text data in a Polars DataFrame.

        Args:
            df (pl.DataFrame): DataFrame containing the text data
            text_column (str, optional): Column name containing the text. Defaults to 'review'.
            length (int, optional): Minimum length of text to summarize. Defaults to 500.
            BATCH_SIZE (int, optional): Number of samples to process in each batch. Defaults to 32.

        Returns:
            pl.DataFrame: Polars DataFrame containing the summarized texts
        """

        df_summary = self._extract_text(df, text_column, length, option='summary')

        batch_inputs = self._create_input_list(df_summary, text_column)
    
        batch_outputs = await self._async_batch_llm_request(
            batch_inputs = batch_inputs, 
            BATCH_SIZE = BATCH_SIZE, 
            llm_chain = self.summarization_chain
            )

        return df_summary.with_columns([
                    pl.Series("Summary", batch_outputs, strict=False),
                    pl.lit("Summarized").alias("Tag")
                    ])
    
    async def async_translate_text(
            self,
            df: pl.DataFrame,
            text_column: str='text',
            length: int=500,
            BATCH_SIZE: int = 32,
            ) -> pl.DataFrame:
        """
        Asynchronously translate text data in a Polars DataFrame with a limited length as output.

        Args:
            df (pl.DataFrame): Polars DataFrame containing the text data
            text_column (str, optional): Column name containing the text. Defaults to 'text'.
            length (int, optional): Minimum length of text to summarize. Defaults to 500.
            BATCH_SIZE (int, optional): Number of samples to process in each batch. Defaults to 32.

        Returns:
            pl.DataFrame: Polars DataFrame containing the translated texts
        """

        df_translate = self._extract_text(df, text_column, length, option='translation')

        batch_inputs = self._create_input_list(df_translate, text_column)

        batch_outputs = await self._async_batch_llm_request(
            batch_inputs = batch_inputs, 
            BATCH_SIZE = BATCH_SIZE, 
            llm_chain = self.translation_chain
            )

        return df_translate.with_columns([
                    pl.Series("Summary", batch_outputs, strict=False),
                    pl.lit("Translated").alias("Tag")
                    ])
    
    async def async_full_translate_text(
            self,
            df: pl.DataFrame,
            text_column: str='review',
            BATCH_SIZE: int = 32,
            ) -> pl.DataFrame:                
        """
        Asynchronously translate text data in a Polars DataFrame with full length as output.

        Args:
            df (pl.DataFrame): Polars DataFrame containing the text data
            text_column (str, optional): Column name containing the text. Defaults to 'text'.
            BATCH_SIZE (int, optional): Number of samples to process in each batch. Defaults to 32.

        Returns:
            pl.DataFrame: Polars DataFrame containing the translated texts
        """

        batch_inputs = self._create_input_list(df, text_column)

        batch_outputs = await self._async_batch_llm_request(
            batch_inputs = batch_inputs, 
            BATCH_SIZE = BATCH_SIZE, 
            llm_chain = self.full_translation_chain
            )

        return df.with_columns([
                    pl.Series("Translation", batch_outputs, strict=False),
                    ])

    async def async_processing(
        self,
        df: pl.DataFrame,
        text_column: str='text',
        BATCH_SIZE: int = 32,
        length: int=500
        ) -> pl.DataFrame:
        """
        Get the summarized and translated texts based on length limit and concatenate the results.
        
        Args:
            df (pl.DataFrame): Polars DataFrame containing the text data
            text_column (str, optional): Column name containing the text. Defaults to 'text'.
            BATCH_SIZE (int, optional): Number of samples to process in each batch. Defaults to 32.
            length: Maximum length of text to summarize, text with less will only be translated.

        Returns:
            Polars DataFrame with output from both functions and tag
        """
        
        df_summary = await self.async_summarize_text(df, text_column, length, BATCH_SIZE)
        df_translate = await self.async_translate_text(df, text_column, length, BATCH_SIZE)

        return pl.concat(
            [df_summary, df_translate], 
            how="vertical"
            )