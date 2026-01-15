import asyncio
import polars as pl

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

from workflow.services.sentiment_analyzer import SentimentAnalyzer
from workflow.utils.data_reader import read_csv_file
from workflow.utils.database import save_to_duckdb, show_agg_view
from workflow.utils.timer import log_time

@log_time
async def workflow(
    input_file_path: str,
    nrows: int,
    batch_size: int,
    text_column: str,
    schema_overrides: dict = None,
    ) -> pl.DataFrame:
    """
    Main function for running the SentimentAnalyzer.
    """
    
    schema_overrides = {"ID": pl.String}

    data = read_csv_file(
        file_path=input_file_path, 
        nrows=nrows,
        schema_overrides=schema_overrides
        )

    sentiment_analyzer = SentimentAnalyzer(max_retries=1)

    return await sentiment_analyzer.async_analyze_sentiment(
        df=data, 
        text_column=text_column,
        BATCH_SIZE=batch_size,
        )

@app.command()
def main(
    input_file_path: str = typer.Option(
                    "20newsgroups_sci_med.csv", 
                    help="Name of the file that is to be loaded."
                            ),
    nrows: int  = typer.Option(
                    128, 
                    help="Number of rows that is to be loaded from the file."
                            ),
    batch_size: int  = typer.Option(
                    32, 
                    help="Number of rows that is to be processed in each batch."
                            ),
    text_column: str = typer.Option(
                    "text", 
                    help="Name of the text column that is to be analyzed."
                            ),
    output_table: str = typer.Option(
                    "result_sentiment_analysis",
                    help="Name of the DuckDB table to save the analyzed sentiments."
                    ),
    ) -> pl.DataFrame:
    """
    Entry point for executing the workflow.
    """
    # Run the workflow asynchronously and get the DataFrame as result
    df = asyncio.run(
        workflow(
            input_file_path=input_file_path,
            nrows=nrows,
            batch_size=batch_size,
            text_column=text_column,
        )
    )
    
    # Ensure correct data types before saving to DuckDB
    df_output = df.with_columns([
                    pl.col("ID").cast(pl.Utf8),
                    pl.col("Sentiment").cast(pl.String),
                    ])

    # Save the output DataFrame to DuckDB
    save_to_duckdb(
        df=df_output,
        table_name=output_table,
        )
    
    df_output.show()

    show_agg_view(table_name=output_table).show()

    typer.echo(f"Results of sentiments analysis saved to DuckDB: {output_table}")
    
if __name__ == "__main__":
    # Ensure NO_PROXY is set for localhost connections
    import os
    os.environ["NO_PROXY"] = "localhost, 127.0.0.1"
    os.environ["no_proxy"] = "localhost, 127.0.0.1"

    app()