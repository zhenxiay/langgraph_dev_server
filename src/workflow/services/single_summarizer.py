"""This module provides a class with langchain framework to summarize large datasets with single API calls."""
import asyncio
import logging
from typing import List

import numpy as np
import pandas as pd
from langchain_core.output_parsers import StrOutputParser
from tqdm.auto import tqdm
from utils.llm import get_azure_openai_llm
from utils.prompt_template import get_summarization_prompt, get_translation_prompt

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
        model: str = "gpt-4.1",
        temperature: float = 0
    ):
        """Initialize the batch summarizer.
        
        Args:
            model: Model name (e.g., "gpt-4o-mini", "gpt-4", "gpt-3.5-turbo")
            temperature: LLM temperature (0 = deterministic)
        """
        self.llm = get_azure_openai_llm(
            model=model,
            temperature=temperature,
            max_retries=2
        )

        # Create chains using RunnableSequence (pipe operator)
        self.summarization_chain = get_summarization_prompt() | self.llm | StrOutputParser()
        logger.info(f"Excel Notes Summarizer initialized with model: {model}")

        self.translation_chain = get_translation_prompt() | self.llm | StrOutputParser()
        logger.info(f"Excel Notes Translator initialized with model: {model}")
    
    def count_words(self, text: str) -> int:
        """Count the number of words in a text string.
        
        Args:
            text: Input text
            
        Returns:
            Number of words
        """
        if pd.isna(text) or not isinstance(text, str):
            return 0
        return len(str(text).split())
    
    async def summarize_text(self, text: str) -> str:
        """Summarize a single text using the LLM chain.
        
        Args:
            text: Text to summarize
            
        Returns:
            Summarized text
        """
        try:
            result = await self.summarization_chain.ainvoke({"text": text})
            return result.strip()
        except Exception as e:
            logger.error(f"Error summarizing text: {str(e)}")
            return f"Error: {str(e)}"
        
    async def translate_text(self, text: str) -> str:
        """Translate a single text using the LLM chain.
        
        Args:
            text: Text to translate
            
        Returns:
            Summarized text
        """
        try:
            result = await self.summarization_chain.ainvoke({"text": text})
            return result.strip()
        except Exception as e:
            logger.error(f"Error translating text: {str(e)}")
            return f"Error: {str(e)}"

    async def process_batch(
        self, 
        batch_df: pd.DataFrame, 
        notes_column: str, 
        min_words: int
        ) -> List[dict]:
        """Process a batch of rows.
        
        Args:
            batch_df: DataFrame batch to process
            notes_column: Column name containing notes
            min_words: Minimum words for summarization
            
        Returns:
            List of results with summary and tag
        """
        results = []
        
        for _, row in batch_df.iterrows():
            notes = row[notes_column]
            word_count = self.count_words(notes)
            
            if word_count > min_words:
                summary = await self.summarize_text(notes)
                tag = 'Summarized'
            else:
                summary = await self.translate_text(notes)
                tag = 'Translated Only'
            
            results.append({
                'summary': summary,
                'tag': tag,
                'word_count': word_count
            })
        
        return results
    
    async def process_data_async(
        self,
        df: pd.DataFrame,
        notes_column: str = "Notes",
        min_words: int = 25,
        batch_size: int = 32
    ):
        """Process an Excel file and add summarizations.
        
        Args:
            notes_column: Name of the column containing notes
            min_words: Minimum number of words required for summarization
        """
        try:
            # Check if the Notes column exists
            if notes_column not in df.columns:
                raise ValueError(f"Column '{notes_column}' not found in Excel file. Available columns: {df.columns.tolist()}")
            
            # Create new columns for summaries
            df["Summary"] = ""
            df["Tag"] = ""
            
            # Create variables for progress tracking
            total_rows = len(df)
            num_batches = int(np.ceil(total_rows / batch_size))
            summarized_count = 0
            translated_count = 0

            # Process in batches with progress bar
            with tqdm(total=num_batches, desc="Processing batches", unit="batch", ncols=100) as pbar_batch:
                for batch_idx in range(num_batches):
                    start_idx = batch_idx * batch_size
                    end_idx = min(start_idx + batch_size, total_rows)
                    
                    # Get batch
                    batch_df = df.iloc[start_idx:end_idx]
                    
                    # Process batch
                    results = await self.process_batch(batch_df, notes_column, min_words)
                    
                    # Update dataframe
                    for i, result in enumerate(results):
                        row_idx = start_idx + i
                        df.at[row_idx, "Summary"] = result['summary']
                        df.at[row_idx, "Tag"] = result['tag']
                        
                        if result['tag'] == 'Summarized':
                            summarized_count += 1
                        else:
                            translated_count += 1
                    
                    # Update progress bar
                    pbar_batch.update(1)
                    pbar_batch.set_postfix({
                        'rows': f"{end_idx}/{total_rows}",
                        'summarized': summarized_count,
                        'translated': translated_count
                    })
            
            logger.info("Processing complete!")
            logger.info(f"Summarized: {summarized_count}, Translated: {translated_count}")
            
            return df

        except Exception as e:
            logger.error(f"Error processing DataFrame: {str(e)}")
            raise

    def arun_process(
        self,
        df: pd.DataFrame,
        notes_column: str = "Notes",
        min_words: int = 25,
        batch_size: int = 32
        ) -> pd.DataFrame:
        """Synchronous wrapper for async processing.
        
        Args:
            df: Input DataFrame to process
            notes_column: Column containing notes
            min_words: Minimum words for summarization
            batch_size: Rows per processing batch
            
        Returns:
            DataFrame with summaries added
        """
        try:
            return asyncio.run(
                self.process_data_async(df, notes_column, min_words, batch_size)
            )
        except Exception as e:
            logger.error(f"Error while running the process: {str(e)}")
            raise