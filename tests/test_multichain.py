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

    # 1. Your existing chains
    translation_chain = batch_summarizer.NotesSummarizer().full_translation_chain
    sentiment_chain = sentiment_analyzer.SentimentAnalyzer().sentiment_analysis_chain

    # 2. Connect them with a "bridge" 
    # We take the output of chain 1 (x) and wrap it for chain 2
    full_chain = translation_chain | (lambda x: {"text": x}) | sentiment_chain

    # Combine the chains into a parallel structure
    parallel_chain = RunnableParallel({
        "translation": translation_chain,
        "sentiment": sentiment_chain,
    })

    # 3. Run it
    result = parallel_chain.invoke({"text": "Das ist fantastisch!"})

    print("Multi-chain Workflow Result:")
    print(result)

if __name__ == "__main__":
    test_multichain_workflow()