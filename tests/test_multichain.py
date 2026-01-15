"""
This script tests multi-chain workflows.    
"""
import polars as pl

import asyncio
import sys
from pathlib import Path

src_path = Path().resolve() / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from langchain_core.runnables import RunnableParallel

from workflow.utils.data_reader import read_csv_file
from workflow.utils.timer import log_time
from workflow.services import batch_summarizer
from workflow.services import sentiment_analyzer

from dotenv import load_dotenv
load_dotenv()

# Ensure NO_PROXY is set for localhost connections
import os
os.environ["NO_PROXY"] = "localhost, 127.0.0.1"
os.environ["no_proxy"] = "localhost, 127.0.0.1"

def test_multichain_workflow():
    """
    Main function for testing multi-chain workflows.
    """

    translation_chain = batch_summarizer.NotesSummarizer().full_translation_chain
    sentiment_chain = sentiment_analyzer.SentimentAnalyzer().sentiment_analysis_chain

    # We take the output of chain 1 (x) and wrap it for chain 2
    full_chain = translation_chain | (lambda x: {"text": x}) | sentiment_chain

    # Combine the chains into a parallel structure
    parallel_chain = RunnableParallel({
        "translation": translation_chain,
        "sentiment": sentiment_chain,
    })

    data = read_csv_file(file_path="C:/Users/YUZ1KA/Downloads/crm_text_collection.csv", nrows=32)
    input_text = sentiment_analyzer.SentimentAnalyzer()._create_input_list(data, text_column='Notes')

    result = parallel_chain.batch(input_text)

    output_data = data.with_columns(
        pl.Series("Translation", [res["translation"] for res in result]),
        pl.Series("Sentiment", [res["sentiment"] for res in result])
    )

    print("Multi-chain Workflow Result:")
    print(output_data.show())

if __name__ == "__main__":
    test_multichain_workflow()