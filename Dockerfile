# Dockerfile for running LangGraph local server (multi-stage build)
# Python >=3.11 is required
# Steps following docs uinder: https://docs.langchain.com/langgraph-platform/local-server#4-create-a-env-file

FROM python:3.11-slim
WORKDIR /app

# Proxy envs (if needed)
ENV http_proxy=""
ENV https_proxy=""

# Install the LangGraph CLI
RUN pip install -U "langgraph-cli[inmem]"

# Create a LangGraph app
RUN langgraph new /app/langgraph_server --template new-langgraph-project-python

# Install dependencies
RUN cd /app/langgraph_server && pip install -e .

# Copy .env file (ensure you have LANGSMITH_API_KEY set)
COPY .env /app/.env

# Copy current state of code to override the initial code in the container
COPY src/ /app/langgraph_server/src

# Expose the default port
EXPOSE 2024

# Start the LangGraph server
RUN cd /app/langgraph_server
CMD ["langgraph", "dev", "--config", "/app/langgraph_server/langgraph.json"]
