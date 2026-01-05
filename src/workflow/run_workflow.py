import asyncio
import pandas as pd

import sys
from pathlib import Path

src_path = Path().resolve() / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from dotenv import load_dotenv
load_dotenv()

# Config typer app
import typer
app = typer.Typer()

from workflow.services.batch_summarizer import NotesSummarizer

async def workflow(
    input_file_path: str,
    nrows: int ,
    text_column: str,
    length: int,
    ) -> pd.DataFrame:
    """Main function for testing the NotesSummarizer."""
    test_data = pd.read_csv(
        input_file_path, 
        nrows=nrows, 
        delimiter=';'
        )
    text_summarizer = NotesSummarizer()

    return await text_summarizer.async_processing(
        df=test_data, 
        text_column=text_column,
        length=length
        )

@app.command()
def main(
    input_file_path: str = typer.Option(
                    "20newsgroups_sci_med.csv", 
                    help="Name of the file that is to be loaded."
                            ),
    nrows: int  = typer.Option(
                    128, 
                    help="Number of rows that is to be processed."
                            ),
    text_column: str = typer.Option(
                    "text", 
                    help="Name of the text column that is to be summarized."
                            ),
    length: int = typer.Option(
                    100, 
                    help="Length limit of the text that is to be summarized by LLM."
                            ),
    output_path: str = typer.Option(
                    "batch_summarized_notes.xlsx",
                    help="Path to save the summarized notes."
                    ),
    ) -> pd.DataFrame:
    """Entry point for executing the workflow."""

    df_output = asyncio.run(
        workflow(
            input_file_path=input_file_path,
            nrows=nrows,
            text_column=text_column,
            length=length,
        )
    )

    df_output.to_excel(
        output_path, 
        index=False
        )
    
    typer.echo(f"Summarized notes saved to: {output_path}")
    
if __name__ == "__main__":
    # Ensure NO_PROXY is set for localhost connections
    import os
    os.environ["NO_PROXY"] = "localhost, 127.0.0.1"
    os.environ["no_proxy"] = "localhost, 127.0.0.1"

    app()