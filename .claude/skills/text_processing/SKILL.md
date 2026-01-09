---
name: test-processing-guide
description: "Comprehensive guide for AI agent when it is requested to perform following tasks: (1) Summarize text, (2) Sentiment analysis, (3) Translate test in full length."
---

# DOCX creation, editing, and analysis

## Overview

A user may ask you to summarize text from a file, run sentiment analysis of text or translate text in full length. You have different tools and workflows available for different tasks.

## Workflow Decision Tree

### Summarize text
Use "Text summary" workflow

### Sentiment analysis
Use "Sentiment analysis" workflow

### Translate text in full length
Use "Translate text" workflow

## Workflow list

### Text summary

1. Ask user for necessary inputs:
   - input_file_path
   - nrows
   - text_column
   - length
   - output_path

   If user needs help for some of the input parameters, run following command to get help:
   ```
   uv run src/workflow/run_workflow.py --help
   ```

2. Run following command to finish the requested task:

   ```
   uv run src/workflow/run_workflow.py
   ```

### Sentiment analysis

1. Ask user for necessary inputs:
   - input_file_path
   - nrows
   - text_column
   - output_path

   If user needs help for some of the input parameters, run following command to get help:
   ```
   uv run src/workflow/run_sentiment_analysis.py --help
   ```

2. Run following command to finish the requested task:

   ```
   uv run src/workflow/run_sentiment_analysis.py
   ```

### Translate text

1. Ask user for necessary inputs:
   - input_file_path
   - nrows
   - text_column
   - output_path

   If user needs help for some of the input parameters, run following command to get help:
   ```
   uv run src/workflow/run_full_translation.py --help
   ```

2. Run following command to finish the requested task:

   ```
   uv run src/workflow/run_full_translation.py
   ```

## Error handling

**Script cannot be found**: If the Agent gets this error, use `powershell`commands to search for the script on the local machine.

**Parameters missing or incorrect**: If the Agent gets error regarding any missing or incorrect input parameters, repeat *step 1* and confirm with user.