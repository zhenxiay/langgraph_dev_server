"""This module provides a class with langchain framework to summarize large datasets with single API calls."""
import asyncio
import logging
from typing import List

import numpy as np
import pandas as pd
from langchain_core.output_parsers import StrOutputParser
from tqdm.auto import tqdm
from workflow.utils.llm import get_azure_openai_llm, get_openai_llm
from workflow.utils.prompt_template import (
    get_summarization_prompt, 
    get_translation_prompt
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Disable httpx INFO logs
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)

class NotesSummarizer:
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
        if provider == "AzureOpenAI":
            self.llm = get_azure_openai_llm(
                model=model,
                temperature=temperature,
                max_retries=2
            )
        else:
            self.llm = get_openai_llm(
                model=model,
                temperature=temperature,
                max_retries=2
            )

        # Create chains using RunnableSequence (pipe operator)
        self.summarization_chain = get_summarization_prompt() | self.llm | StrOutputParser()
        logger.info(f"Excel Notes Summarizer initialized with provider-model: {provider}-{model}")

        self.translation_chain = get_translation_prompt() | self.llm | StrOutputParser()
        logger.info(f"Excel Notes Translator initialized with provider-model: {provider}-{model}")    
    
    async def async_summarize_text(
            self,
            df: pd.DataFrame,
            text_column: str='review',
            length: int=500
            ) -> pd.DataFrame:

        df['review_length'] = df[text_column].apply(lambda x: len(x))

        df_summary = df[df['review_length'] > length]

        text_list = [text for text in df_summary[text_column].tolist()]
        batch_inputs = []
        batch_outputs = []

        for text in text_list:
            batch_inputs.append({"text": text})
    
        batch_outputs = await self.summarization_chain.abatch(batch_inputs)

        df_summary=df_summary.assign(Summary=batch_outputs).assign(Tag='Summarized')

        return df_summary
    
    async def async_tanslate_text(
            self,
            df: pd.DataFrame,
            text_column: str='review',
            length: int=500
            ) -> pd.DataFrame:

        df['review_length'] = df[text_column].apply(lambda x: len(x))

        df_translate = df[df['review_length'] <= length]

        text_list = [text for text in df_translate[text_column].tolist()]
        batch_inputs = []
        batch_outputs = []

        for text in text_list:
            batch_inputs.append({"text": text})

        batch_outputs = await self.translation_chain.abatch(batch_inputs)

        df_translate=df_translate.assign(Summary=batch_outputs).assign(Tag='Translated')
    
        return df_translate


    async def async_processing(
        self,
        df: pd.DataFrame,
        text_column: str='review',
        length: int=500
        ) -> pd.DataFrame:
        """Process a batch of rows asynchronously.
        
        Args:
            length: Maximum length of text to summarize, text with less will only be translated.

        Returns:
            Pandas Dataframe with summary and tag
        """
        
        df_summary = await self.async_summarize_text(df, text_column, length)
        df_translate = await self.async_tanslate_text(df, text_column, length)

        return pd.concat(
            [df_summary, df_translate], 
            ignore_index=True
            )