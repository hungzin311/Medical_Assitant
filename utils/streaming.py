from contextvars import ContextVar
from typing import Optional

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import AIMessage


current_stream_callback: ContextVar[Optional[BaseCallbackHandler]] = ContextVar(
    "current_stream_callback",
    default=None,
)


def get_stream_config():
    callback = current_stream_callback.get()
    if callback is None:
        return None
    return {"callbacks": [callback]}


def emit_stream_event(event: str, payload: dict):
    callback = current_stream_callback.get()
    if callback is not None and hasattr(callback, "emit_event"):
        callback.emit_event(event, payload)


def invoke_with_streaming(model, prompt, max_completion_token = 1024):
    extra_body = {
        "chat_template_kwargs": {"enable_thinking": False},
        "max_completion_tokens": max_completion_token
    }
    callback = current_stream_callback.get()
    if callback is None:
        return  model.bind(extra_body= extra_body).invoke(prompt)


    content_parts = []
    for chunk in model.bind(extra_body= extra_body).stream(prompt):
        chunk_content = getattr(chunk, "content", "")
        if chunk_content:
            content_parts.append(chunk_content)
            callback.on_llm_new_token(chunk_content)

    return AIMessage(content="".join(content_parts))
