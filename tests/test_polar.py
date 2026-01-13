"""
This script tests the data reading utilities using Polars library.    
"""
import polars as pl

import sys
from pathlib import Path

src_path = Path().resolve() / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from workflow.utils.data_reader import read_csv_file

def test_process_csv():

    # Define the path to the sample CSV file
    current_dir = Path(__file__).parent.parent
    sample_file_path = current_dir / "20newsgroups_sci_med.csv"

    # Read the CSV file using the utility function
    df = read_csv_file(file_path=str(sample_file_path), nrows=128)

    df_with_length = df.with_columns(
            pl.col("text").str.len_chars().alias("text_length")
        )
    
    return df_with_length

if __name__ == "__main__":
    df_result = test_process_csv()
    print(df_result.head())