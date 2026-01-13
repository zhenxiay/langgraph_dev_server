---
name: text-summary-guide
description: Guide for AI agent to perform text summary task.
---

# Text Summary Skill Guide

## Overview

This file is a guide for AI agent to perform text summary task.

---

# When to use this skill
This skill is to be loaded when an AI agent is requested to perform **text summary** task.

## Workflow

1. Ask user for necessary inputs:
   - input_file_path
   - nrows
   - batch_size
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

## Error handling

**Script cannot be found**: If the Agent gets this error, use `powershell`commands to search for the script on the local machine.

**Parameters missing or incorrect**: If the Agent gets error regarding any missing or incorrect input parameters, repeat *step 1* and confirm with user.