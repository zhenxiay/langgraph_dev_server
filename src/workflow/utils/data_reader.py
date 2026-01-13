"""This module provides utility functions to read data files into pandas DataFrames."""
import logging
import polars as pl

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def read_excel_file(file_path: str, nrows: int = 128) -> pl.DataFrame:
    """Reads an Excel file and returns a DataFrame.

    Args:
        file_path (str): The path to the Excel file.
        nrows (int): The number of rows to read from the file. Default is 128.
    """
    try:
        df = pl.read_excel(
            file_path, 
            n_rows=nrows
            )
        logger.info(f"Successfully read the Excel file: {file_path}")
        logger.info(f"Rows loaded: {len(df)}")
        return df
    except Exception as e:
        logger.error(f"Error reading the Excel file: {e}")
        raise

def read_csv_file(file_path: str, nrows: int = 128) -> pl.DataFrame:
    """Reads a CSV file and returns a DataFrame.

    Args:
        file_path (str): The path to the csv file.
        nrows (int): The number of rows to read from the file. Default is 128.
    """
    try:
        df = pl.read_csv(
                file_path, 
                n_rows=nrows,
                separator=';'
                )
        logger.info(f"Successfully read the CSV file: {file_path}")
        logger.info(f"Rows loaded: {len(df)}")
        return df
    except Exception as e:
        logger.error(f"Error reading the CSV file: {e}")
        raise