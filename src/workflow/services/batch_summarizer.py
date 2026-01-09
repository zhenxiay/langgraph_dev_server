"""This module provides a class with langchain framework to summarize large datasets with single API calls."""
import asyncio
import logging
from typing import List

import numpy as np
import pandas as pd
from langchain_core.output_parsers import StrOutputParser
from langchain_core.tracers.schemas import Run
from workflow.utils.llm import BaseLLMService
from workflow.utils.prompt_template import (
    get_summarization_prompt, 
    get_translation_prompt,
    get_full_translation_prompt,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Disable httpx INFO logs
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)

class NotesSummarizer(BaseLLMService):
    """Summarizes notes from Excel files using OpenAI and LangChain."""
    
    def __init__(
        self,
        provider: str = "AzureOpenAI",
        model: str = "gpt-4.1",
        temperature: float = 0
    ):
        """Initialize the batch summarizer.
        
        Args:
            provider: LLM provider ("AzureOpenAI" or "OpenAI")
            model: Model name (e.g., "gpt-4o-mini", "gpt-4", "gpt-3.5-turbo")
            temperature: LLM temperature (0 = deterministic)
        """
        super().__init__(provider, model, temperature)

        # Define max concurrent requests for batch processing
        self.chain_config = {"max_concurrent_requests": 512}

        # Create chains using RunnableSequence (pipe operator)
        self.summarization_chain = get_summarization_prompt() | self.llm | StrOutputParser()
        logger.info(f"Excel Notes Summarizer initialized with provider-model: {provider}-{model}")

        self.translation_chain = get_translation_prompt() | self.llm | StrOutputParser()
        logger.info(f"Excel Notes Translator initialized with provider-model: {provider}-{model}")

        self.full_translation_chain = get_full_translation_prompt() | self.llm | StrOutputParser()
        logger.info(f"Excel Notes Full Translator initialized with provider-model: {provider}-{model}")    
    
    def _extract_text(
            self,
            df: pd.DataFrame,
            text_column: str='review',
            length: int=500,
            option: str='summary'
            ) -> pd.DataFrame:
        """
        Extract summary texts from DataFrame based on given length limit.
        """

        df['text_length'] = df[text_column].str.len()

        if option == 'summary':
            return df[df['text_length'] > length]
        elif option == 'translation':
            return df[df['text_length'] <= length]
    
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

    async def on_start_summarizer(self, run_obj: Run):
        """Callback function to log when summarization starts."""
        
        logger.info(f"Summarization started for ID: {run_obj.id}")

    async def on_end_summarizer(self, run_obj: Run):
        """Callback function to log when summarization ends."""

        duration = run_obj.end_time - run_obj.start_time

        logger.info(f"Task {run_obj.id} finished in {duration.total_seconds():.2f}s")
        
    async def async_summarize_text(
            self,
            df: pd.DataFrame,
            text_column: str='review',
            length: int=500
            ) -> pd.DataFrame:
        """
        Asynchronously summarize text data in a DataFrame.

        Args:
            df (pd.DataFrame): DataFrame containing the text data
            text_column (str, optional): Column name containing the text. Defaults to 'review'.
            length (int, optional): Minimum length of text to summarize. Defaults to 500.

        Returns:
            pd.DataFrame: DataFrame containing the summarized texts
        """

        df_summary = self._extract_text(df, text_column, length, option='summary')

        batch_inputs = self._create_input_list(df_summary, text_column)
    
        batch_outputs = await self.summarization_chain.abatch(batch_inputs, return_exceptions=True)

        return df_summary.assign(Summary=batch_outputs)\
                         .assign(Tag='Summarized')
    
    async def async_translate_text(
            self,
            df: pd.DataFrame,
            text_column: str='review',
            length: int=500
            ) -> pd.DataFrame:

        df_translate = self._extract_text(df, text_column, length, option='translation')

        batch_inputs = self._create_input_list(df_translate, text_column)

        batch_outputs = await self.translation_chain.abatch(batch_inputs, return_exceptions=True)

        return df_translate.assign(Translation=batch_outputs)\
                           .assign(Tag='Translated')
    
    async def async_full_translate_text(
            self,
            df: pd.DataFrame,
            text_column: str='review',
            ) -> pd.DataFrame:

        batch_inputs = self._create_input_list(df, text_column)

        batch_outputs = await self.full_translation_chain\
                                        .with_alisteners(
                                            on_start=self.on_start_summarizer,
                                            on_end=self.on_end_summarizer
                                            ) \
                                        .abatch(
                                            batch_inputs, 
                                            return_exceptions=True, 
                                            config=self.chain_config
                                            )

        return df.assign(Summary=batch_outputs)

    async def async_processing(
        self,
        df: pd.DataFrame,
        text_column: str='review',
        length: int=500
        ) -> pd.DataFrame:
        """
        Process a batch of rows asynchronously.
        
        Args:
            length: Maximum length of text to summarize, text with less will only be translated.

        Returns:
            Pandas Dataframe with summary and tag
        """
        
        df_summary = await self.async_summarize_text(df, text_column, length)
        df_translate = await self.async_translate_text(df, text_column, length)

        return pd.concat(
            [df_summary, df_translate], 
            ignore_index=True
            )