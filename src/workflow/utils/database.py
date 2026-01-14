"""
This module contains utility functions for database which are used to store output data.
"""
import duckdb
import polars as pl

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
    conn = duckdb.connect(duckdb_path)

    # Save DataFrame to DuckDB table
    query = f"CREATE TABLE IF NOT EXISTS {table_name} AS SELECT * FROM df"
    
    conn.execute(query)

    conn.close()
