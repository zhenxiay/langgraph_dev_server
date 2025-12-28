# NotesSummarizer API Reference

Complete reference for programmatic usage of the text summarization service.

## Class: NotesSummarizer

Main class providing text summarization and translation capabilities with async processing.

### Constructor

```python
from workflow.services import single_summarizer

summarizer = single_summarizer.NotesSummarizer(
    model="gpt-4.1",
    temperature=0
)
```

**Parameters:**
- `model` (str): OpenAI model name (default: "gpt-4.1")
- `temperature` (float): Generation randomness 0-1 (default: 0)

### Core Methods

#### `async_processing(df, text_column, length)`

Primary method for batch text processing with intelligent routing.

```python
result_df = await summarizer.async_processing(
    df=dataframe,
    text_column="content", 
    length=500
)
```

**Parameters:**
- `df` (pd.DataFrame): Input data containing text
- `text_column` (str): Column name with text content (default: "review")
- `length` (int): Word threshold for summarization vs translation (default: 500)

**Returns:**
- `pd.DataFrame`: Original data with added columns:
  - `Summary`: Generated summary or translation
  - `Tag`: Processing type ("Summarized" or "Translated")

**Processing Logic:**
- Text > `length` words ? Summarization
- Text = `length` words ? Translation (Chinese to English)

#### `async_summarize_text(df, text_column, length)`

Direct summarization for texts above length threshold.

```python
summary_df = await summarizer.async_summarize_text(
    df=long_texts_df,
    text_column="content",
    length=500
)
```

#### `async_translate_text(df, text_column, length)`

Direct translation for texts below length threshold.

```python
translated_df = await summarizer.async_translate_text(
    df=short_texts_df,
    text_column="content", 
    length=500
)
```

### Utility Methods

#### `summarize_text(text)`

Single text summarization (async).

```python
summary = await summarizer.summarize_text("Long text content...")
```

#### `translate_text(text)`

Single text translation (async).

```python
translation = await summarizer.translate_text("Short Chinese text")
```

#### `count_words(text)`

Word counting utility.

```python
word_count = summarizer.count_words("Some text content")
```

## Usage Patterns

### Basic DataFrame Processing

```python
import pandas as pd
from workflow.services import single_summarizer

# Load data
df = pd.read_csv("data.csv")

# Initialize
summarizer = single_summarizer.NotesSummarizer()

# Process
result = await summarizer.async_processing(
    df=df,
    text_column="content"
)
```

### Custom Configuration

```python
# High creativity model
summarizer = single_summarizer.NotesSummarizer(
    model="gpt-4o-mini",
    temperature=0.7
)

# Custom length threshold
result = await summarizer.async_processing(
    df=df,
    text_column="reviews",
    length=1000  # Longer threshold
)
```

### Error Handling

```python
try:
    result = await summarizer.async_processing(df=df, text_column="text")
except KeyError as e:
    print(f"Column not found: {e}")
except Exception as e:
    print(f"Processing error: {e}")
```

## Performance Notes

- Uses async batch processing for API efficiency
- Automatically handles API rate limits with retries
- Progress tracking via tqdm for large datasets
- Memory efficient for DataFrames up to ~100k rows

## Model Support

**Recommended models:**
- `gpt-4.1`: Best quality, higher cost
- `gpt-4o-mini`: Good balance of speed/quality
- `gpt-3.5-turbo`: Fastest, lowest cost

**Temperature guidelines:**
- `0`: Deterministic output (default)
- `0.3-0.7`: Balanced creativity
- `0.8-1.0`: High creativity/variety
