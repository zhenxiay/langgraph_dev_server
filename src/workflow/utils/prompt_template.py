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

def get_sentiment_analysis_prompt() -> PromptTemplate:
    """Returns a PromptTemplate for sentiment analysis."""
    return PromptTemplate(
            input_variables=["text"],
            template="""You are an expert in sentiment analysis. 
Your task is to classify the text that customer wrote to following categories: Positive, Very Positive, Neutral, Mixed, Negative, Very Negative.
Provide only the category as output without any explanation.

Text: {text}

Sentiment:"""
        )