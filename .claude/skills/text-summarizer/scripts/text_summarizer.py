#!/usr/bin/env python3
"""
Advanced text summarization script with command-line interface.
Supports batch processing with intelligent routing based on text length.
"""
import argparse
import asyncio
import os
import sys
from pathlib import Path

import pandas as pd

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()
print("✅ Loaded .env file")

def setup_path():
    """Add src directory to Python path."""
    # Get the project root (langgraph_dev_server)
    project_root = Path.cwd()
    src_path = project_root / "src"
    
    if src_path.exists() and str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))
        print(f"✅ Added to Python path: {src_path}")

def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Intelligent text summarization with async processing")
    
    parser.add_argument("--input", "-i", required=True, help="Input file path (CSV or Excel)")
    parser.add_argument("--output", "-o", default="summary_results.csv", help="Output file path")
    parser.add_argument("--text-column", default="text", help="Name of the text column to process")
    parser.add_argument("--length", type=int, default=500, help="Length threshold for summarization vs translation")
    parser.add_argument("--model", default="gpt-4.1", help="OpenAI model to use")
    parser.add_argument("--temperature", type=float, default=0, help="Model temperature")
    parser.add_argument("--max-rows", type=int, help="Limit processing to N rows")
    
    return parser.parse_args()

def load_data(file_path: str, max_rows: int = None) -> pd.DataFrame:
    """Load data from CSV or Excel file."""
    path = Path(file_path)
    
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {file_path}")
    
    if path.suffix.lower() == ".csv":
        df = pd.read_csv(file_path, nrows=max_rows)
    elif path.suffix.lower() in [".xlsx", ".xls"]:
        df = pd.read_excel(file_path, nrows=max_rows)
    else:
        raise ValueError(f"Unsupported file format: {path.suffix}")
    
    return df

async def main():
    """Main execution function."""
    args = parse_arguments()
    
    print("=== Text Summarizer Script ===")
    setup_path()
    
    try:
        # Check for OpenAI API key
        if not os.getenv("OPENAI_API_KEY"):
            print("❌ Error: OPENAI_API_KEY not found!")
            print("📋 Please check:")
            print("   - Your .env file contains: OPENAI_API_KEY=your-key-here")
            return
        else:
            api_key_preview = os.getenv("OPENAI_API_KEY")
            print(f"🔑 OpenAI API key loaded")
        
        # Import after path setup
        print("📦 Importing workflow modules...")
        from workflow.services import single_summarizer
        print("✅ Successfully imported workflow.services.single_summarizer")
        
        print(f"📂 Loading data from: {args.input}")
        df = load_data(args.input, args.max_rows)
        print(f"✅ Loaded {len(df)} rows")
        
        if args.text_column not in df.columns:
            available_columns = ", ".join(df.columns.tolist())
            raise ValueError(f"Column '{args.text_column}' not found. Available columns: {available_columns}")
        
        print(f"⚙️ Configuration:")
        print(f"   Text column: {args.text_column}")
        print(f"   Length threshold: {args.length} words")
        print(f"   Model: {args.model}")
        
        # Initialize summarizer
        print("🤖 Initializing summarizer...")
        summarizer = single_summarizer.NotesSummarizer(model=args.model, temperature=args.temperature)
        
        # Process data
        print(f"🔄 Processing {len(df)} rows...")
        result_df = await summarizer.async_processing(df=df, text_column=args.text_column, length=args.length)
        
        # Save results
        result_df.to_csv(args.output, index=False)
        
        print(f"✅ Results saved to: {args.output}")
        print(f"📊 Processed {len(result_df)} rows")
        
        # Summary statistics
        if "Tag" in result_df.columns:
            tag_counts = result_df["Tag"].value_counts()
            print("📈 Processing summary:")
            for tag, count in tag_counts.items():
                print(f"   {tag}: {count} rows")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
