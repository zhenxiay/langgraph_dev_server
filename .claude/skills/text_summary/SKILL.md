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
If the Agent gets error that the script cannot be found, use `powershell`commands to search for the script on the local machine.

---