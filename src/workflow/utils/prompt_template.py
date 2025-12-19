"""This module defines prompt templates for text summarization tasks."""
from langchain_core.prompts import PromptTemplate


def get_summarization_prompt() -> PromptTemplate:
    """Returns a PromptTemplate for single text summarization."""
    return PromptTemplate(
            input_variables=["text"],
            template="""Summarize the following text. 
Keep the summary concise and under 15 words.
DO NOT include any personal data in the summary (e.g., names, email, locations).
The summary should be written in English, regardless of input language.

Text: {text}

Summary:"""
        )

def get_translation_prompt() -> PromptTemplate:
    """Returns a PromptTemplate for single text translation."""
    return PromptTemplate(
            input_variables=["text"],
            template="""Identify the language of the following text.
If it is not English, then translate the following text into English language.
Keep the length of the translation under 20 words.
DO NOT include any personal data in the summary (e.g., names, email, locations).

Text: {text}

Summary:"""
        )

def get_batch_summarization_prompt() -> PromptTemplate:
    """Returns a PromptTemplate for batch text summarization."""
    return PromptTemplate(
            input_variables=["texts"],
            template="""Summarize each of the following texts separately. 
For each text, keep the summary concise and under 15 words.
DO NOT include any personal data (e.g., names, email, locations).
All summaries should be in English.

Format your response EXACTLY as shown:
[SUMMARY 1]: summary here
[SUMMARY 2]: summary here
[SUMMARY 3]: summary here
etc.

{texts}
"""
        )

def get_batch_translation_prompt() -> PromptTemplate:
    """Returns a PromptTemplate for batch text translation."""
    return PromptTemplate(
            input_variables=["texts"],
            template="""Summarize each of the following texts separately. 
For each text, keep the summary concise and under 15 words.
DO NOT include any personal data (e.g., names, email, locations).
All summaries should be in English.

Format your response EXACTLY as shown:
[SUMMARY 1]: summary here
[SUMMARY 2]: summary here
[SUMMARY 3]: summary here
etc.

{texts}
"""
        )