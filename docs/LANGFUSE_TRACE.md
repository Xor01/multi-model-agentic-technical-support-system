# Langfuse Trace Status

Required trace fields are wired through `support_agent.observability.tracing.invoke_with_langfuse`: user input, trace ID, project/student metadata, LangGraph activity, final answer, latency, and exceptions can be sent through the Langfuse LangChain callback.

No valid trace URL or screenshot is included. On 2026-09-20, the configured credentials returned HTTP 401 against both `https://cloud.langfuse.com` and `https://us.cloud.langfuse.com`. Claiming a trace would therefore be false.

To complete this deliverable:

1. Put a matching public key, secret key, and host in `.env`.
2. Rebuild/restart `support-agent` so the container receives them.
3. Send a chat completion request through FastAPI.
4. Confirm the trace shows router and node/tool spans in Langfuse.
5. Add the trace URL or screenshot here.

Never commit the real Langfuse secret key.
