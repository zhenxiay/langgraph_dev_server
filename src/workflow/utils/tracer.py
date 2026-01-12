"""
Utility module for setting up OpenInference tracing with OpenAI instrumentation.
"""
from openinference.instrumentation.openai import OpenAIInstrumentor
from opentelemetry.sdk import trace as trace_sdk
from opentelemetry.sdk.trace.export import ConsoleSpanExporter, SimpleSpanProcessor
from openinference.instrumentation import TraceConfig

config = TraceConfig(
        hide_inputs=True,
        hide_outputs=True,
        hide_input_messages=True,
        hide_output_messages=False,
        hide_input_images=True,
        hide_input_text=True,
        hide_output_text=True,
    )

def setup_console_tracing():
    """
    Set up OpenInference tracing with OpenAI instrumentation.
    Output will be directly printed to the console.
    """
   
    tracer_provider = trace_sdk.TracerProvider()
    tracer_provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))

    OpenAIInstrumentor().instrument(tracer_provider=tracer_provider, config=config)