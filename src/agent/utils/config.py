"""
Configuration module for the graph.
"""
import os
from langchain.chat_models import init_chat_model

def init_models():
    '''
    Initialize and return a dictionary of chat models.
    '''
    return {
    "anthropic": init_chat_model("anthropic:claude-3-5-sonnet-20240620"),
    "openai": init_chat_model("openai:gpt-4o"),
    }

def setup_no_proxy():
    '''
    Set NO_PROXY to avoid proxy for localhost connections (important for local MCP server access)
    '''
    os.environ["NO_PROXY"] = "localhost, 127.0.0.1"
    os.environ["no_proxy"] = "localhost, 127.0.0.1"

def setup_local_proxy():
    '''
    Set up local proxy server for the connections
    '''
    os.environ["https_proxy"] = "http://127.0.0.1:3128"
    os.environ["http_proxy"] = "http://127.0.0.1:3128"

def system_prompt():
    '''
    Return a default system prompt.
    '''

    return '''You're a course teaching assistant. Answer the QUESTION based on the CONTEXT from the FAQ database.
              Use only the facts from the CONTEXT when answering the QUESTION.'''
