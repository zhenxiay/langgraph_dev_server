"""
This module contains utility functions for database which are used to read or store data.
"""
import duckdb
import polars as pl

def read_from_duckdb(
    table_name: str, 
    duckdb_path: str = "input.duckdb"
    ) -> pl.DataFrame:
    """
    Reads a DuckDB table into a Polars DataFrame.

    Args:
        table_name (str): The name of the DuckDB table.
        duckdb_path (str): The path to the DuckDB database file.

    Returns:
        pl.DataFrame: A Polars DataFrame containing the data from the DuckDB table.
    """
    # Connect to DuckDB
    conn = duckdb.connect(duckdb_path, read_only=True)

    # Query to read the table
    query = f'SELECT * FROM {table_name}'

    result_df = conn.execute(query).pl()

    conn.close()

    return result_df

def save_to_duckdb(
    df: pl.DataFrame, 
    table_name: str, 
    duckdb_path: str = "output.duckdb"
    ) -> None:
    """
    Saves a Polars DataFrame to a DuckDB table.

    Args:
        df (pl.DataFrame): The Polars DataFrame to be saved.
        table_name (str): The name of the DuckDB table.
        duckdb_path (str): The path to the DuckDB database file. Defaults to in-memory database.
    """
    # Connect to DuckDB 
    conn = duckdb.connect(duckdb_path, read_only=False)

    # Save DataFrame to DuckDB table
    query = f'''
    DROP TABLE IF EXISTS {table_name};
    CREATE TABLE {table_name} AS 
    SELECT COLUMNS(*)::VARCHAR
    FROM df
    '''

    conn.execute(query)

    conn.close()

def show_agg_view(
    table_name: str, 
    duckdb_path: str = "output.duckdb"
    ) -> pl.DataFrame:
    """
    Displays aggregated view of the specified DuckDB table.

    Args:
        table_name (str): The name of the DuckDB table.
        duckdb_path (str): The path to the DuckDB database file. Defaults to in-memory database.

    Returns:
        pl.DataFrame: A Polars DataFrame containing the aggregated view.
    """
    # Connect to DuckDB
    conn = duckdb.connect(duckdb_path)

    # Query to get aggregated view
    query = f'''
    SELECT Sentiment, COUNT(*) AS Count
    FROM {table_name}
    GROUP BY Sentiment
    ORDER BY Count DESC
    '''

    result_df = conn.sql(query).pl()

    conn.close()

    return result_df