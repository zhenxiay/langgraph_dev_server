"""This module serves as the main entry point for executing the text summarization workflow."""
import logging
import os
from pathlib import Path

import pandas as pd
import typer
from dotenv import load_dotenv
from services.batch_summarizer import NotesBatchSummarizer
from services.single_summarizer import NotesSummarizer
from utils.data_reader import read_excel_file

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()

# Config typer app
app = typer.Typer()

@app.command()
def main(
    input_file_name: str = typer.Option(
                    "crm_text_collection.xlsx", 
                    help="Name of the file that is to be loaded."
                            ),
    nrows: int  = typer.Option(
                    128, 
                    help="Number of rows that is to be processed."
                            ),
    batch_mode: bool = typer.Option(
                    True, 
                    help="Decides whether to use batch or single summarization mode."
                            ),
    ) -> pd.DataFrame:
    """Entry point for executing the workflow."""
    # Set NO_PROXY to avoid proxy for localhost connections (important for local MCP server access)
    os.environ["NO_PROXY"] = "localhost, 127.0.0.1"
    os.environ["no_proxy"] = "localhost, 127.0.0.1"

    # Use Path.home() for cross-platform compatibility
    downloads_dir = Path.home() / "Downloads"
    input_file_path = downloads_dir / input_file_name
    output_file = downloads_dir / "summarized_notes.xlsx"

    logger.info(f"Reading {nrows} rows from {input_file_path}")
    df = read_excel_file(file_path=input_file_path, nrows=nrows)
    
    if not batch_mode:
        summarizer = NotesSummarizer(
            temperature=0
        )
        
        logger.info("Processing notes in single summarization mode...")

    else:
        summarizer = NotesBatchSummarizer(
            temperature=0,
            batch_api_size=10  # 10 texts per API call = 50-60% token savings
        )

        logger.info("Processing notes in batch summarization mode...")
    
    output_df = summarizer.arun_process(
            df=df,
            notes_column="Notes",
            min_words=25,
            batch_size=32
    )

    output_df.to_excel(output_file, index=False)

    logger.info(f"Summarized notes saved to: {output_file}")

if __name__ == "__main__":
    app()  # typer app entry point
