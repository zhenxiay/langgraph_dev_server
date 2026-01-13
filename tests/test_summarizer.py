"""
This script tests the sentiment analysis workflow.    
"""
import polars as pl

import asyncio
import sys
from pathlib import Path

src_path = Path().resolve() / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from workflow.utils.data_reader import read_csv_file
from workflow.utils.timer import log_time
from workflow.services.batch_summarizer import NotesSummarizer

from dotenv import load_dotenv
load_dotenv()

# Ensure NO_PROXY is set for localhost connections
import os
os.environ["NO_PROXY"] = "localhost, 127.0.0.1"
os.environ["no_proxy"] = "localhost, 127.0.0.1"

@log_time
def test_notes_summarizer():

    # Define the path to the sample CSV file
    current_dir = Path(__file__).parent.parent
    sample_file_path = current_dir / "20newsgroups_sci_med.csv"

    df = read_csv_file(file_path=str(sample_file_path), nrows=8)

    notes_summarizer = NotesSummarizer()

    df_result = asyncio.run(
                            notes_summarizer.async_processing(
                                df=df,
                                length=800,
                                text_column='text',
                                BATCH_SIZE=4
                                )
                            )

    print("Sentiment Analysis DataFrame:")
    print(df_result.head())

    df_result.write_csv("notes_summarization_output.csv")

    print("Notes summarization results saved to 'notes_summarization_output.csv'")

if __name__ == "__main__":
    test_notes_summarizer()