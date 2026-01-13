"""
This script tests the BaseModel from workflow utils module.    
"""
import polars as pl

import sys
from pathlib import Path

src_path = Path().resolve() / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from workflow.utils.data_reader import read_csv_file
from workflow.utils.timer import log_time
from workflow.utils.llm import BaseModel

from dotenv import load_dotenv
load_dotenv()

# Ensure NO_PROXY is set for localhost connections
import os
os.environ["NO_PROXY"] = "localhost, 127.0.0.1"
os.environ["no_proxy"] = "localhost, 127.0.0.1"

@log_time
def test_base_model_extract_text():

    # Define the path to the sample CSV file
    current_dir = Path(__file__).parent.parent
    sample_file_path = current_dir / "20newsgroups_sci_med.csv"

    df = read_csv_file(file_path=str(sample_file_path), nrows=128)

    base_model = BaseModel()

    # Test summary extraction
    summary_df = base_model._extract_text(
        df=df, 
        text_column='text', 
        length=500, 
        option='summary'
        )
    # assert summary_df.height == 2

    # Test translation extraction
    translation_df = base_model._extract_text(
        df=df, 
        text_column='text', 
        length=500, 
        option='translation'
        )
    # assert translation_df.height == 2
    return summary_df, translation_df

if __name__ == "__main__":

    summary_result, translation_result = test_base_model_extract_text()
    print("Summary DataFrame:")
    print(summary_result.head())
    print("\nTranslation DataFrame:")
    print(translation_result.head())