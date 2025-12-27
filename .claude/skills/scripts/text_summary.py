'''
This script serves as the entry point for the text summarization skill.
'''
import sys
import asyncio
from pathlib import Path

src_path = Path().resolve() / "../../../src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

import pandas as pd
from workflow.services import single_summarizer

async def main():
    '''
    Entry point for text summarization script.
    '''
    
    test_data = pd.read_csv('../20newsgroups_sci_med.csv', nrows=64)

    text_summarizer = single_summarizer.NotesSummarizer()

    result = await text_summarizer.async_processing(
        df=test_data, 
        text_column='text', 
        length=1000
        )
    
    return result

if __name__ == "__main__":
    summary_result = asyncio.run(main())
    summary_result.head().to_csv(
        'text_summary_output.csv', 
        index=False
        )