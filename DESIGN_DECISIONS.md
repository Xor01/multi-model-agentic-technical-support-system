# Design Decisions

## Scope and evidence policy

This is a time-boxed learning system, not a production claim. Missing measurements are recorded as `NOT_EVALUATED`; local mocks and deterministic fallbacks are identified as such. Notebook outputs are reported only where an executed output exists. No loss curve or baseline score is reconstructed.

## Model roles

- **Model A — intent classifier:** predicts one of eight support intents. It is meant to provide the high-confidence middle layer of the router.
- **Model B — extractive QA:** answers from retrieved trusted passages. Retrieval always happens before this model slot.
- **Model C — LoRA support specialist:** synthesizes troubleshooting guidance from the message, retrieved context, and tool results.

The runtime remains available when model dependencies or artifacts are absent. That makes the demo resilient, but it also means a healthy API does not prove that all trained models are loaded. `/ready` exposes the classifier mode for that reason.

## Router policy

The intended production order is:

1. Hard safety and escalation rules.
2. Model A when confidence is at least 0.75.
3. OpenAI `gpt-5-mini` with a strict JSON schema for ambiguous inputs.
4. A safe support-specialist fallback if routing infrastructure fails.

The deployed graph uses `configured_hybrid_route`. If Model A is unavailable, it sends ambiguous requests to `gpt-5-mini` when `OPENAI_API_KEY` is configured and uses deterministic routing if the provider is unavailable. The recorded 400-row evaluation predates this provider integration and used 397 deterministic fallbacks plus three hard rules; it must not be presented as a measured GPT-5 Mini result. The rules/classifier and LLM-only alternatives remain `NOT_EVALUATED` end to end.

## Graph flow

The LangGraph state carries the user message, route, intent, confidence, retrieved context, tool results, answer, escalation flag, and trace ID. Routing selects one branch:

- `qa`: trusted KB retrieval followed by the Model B runner slot.
- `tools`: deterministic diagnostic runbook, then support synthesis.
- `support`: direct support synthesis.
- `escalate`: creates a high-priority support ticket with evidence.

Model B and Model C currently use explicit deterministic fallback runners. This keeps local tests and Open WebUI usable without pretending that the saved weights are serving inference.

## Tool boundary

All 13 requested contracts are registered in `support_agent.tools.ALL_TOOLS` and return dictionaries:

1. `knowledge_base_search`
2. `ticket_search`
3. `ticket_create`
4. `system_health_check`
5. `log_analyzer`
6. `documentation_search`
7. `package_lookup`
8. `sql_query`
9. `calculator`
10. `file_search`
11. `web_search`
12. `escalate_to_human`
13. `diagnostic_runbook`

The lab favors deterministic local implementations. SQL and file access are restricted; the calculator accepts a narrow arithmetic character set; escalations create structured tickets; and the runbook composes health and ticket tools.

## Evaluation gates

A model cannot pass merely because training completed. Each model needs a recorded pre-fine-tuning baseline, required task metrics, and no required Golden Set regression. The current model gates are `NOT_EVALUATED` because baseline records are absent. A revised Golden Set with concrete questions, a synthetic exposed token and ticket, and an injected context passage passed 10/10 on 2026-09-21. Earlier 1/10 and 3/10 runs used different, underspecified prompts; they are retained as history, not compared as equivalent scores. The passing gate covers the current ten scenarios only, not arbitrary production inputs.

The retrieval score is a same-corpus check: each question is searched against the corpus containing its own reference answer. Its perfect score verifies indexing/ranking plumbing but is optimistic and is not an independent generalization result.

## Observability

FastAPI calls the graph through `invoke_with_langfuse`, passing the request trace ID and student/project metadata. Langfuse initialization is optional so absent or invalid credentials do not take down the API. A valid trace should show input, route decision, tool/specialist calls, final answer, latency, and errors.

Current credentials returned HTTP 401 on the default/EU and US Langfuse cloud hosts, so this repository does not claim a trace that was not observed.

## Deployment

The current Compose file exposes FastAPI on host port 8000 and Open WebUI on host port 3000. Open WebUI must call `http://support-agent:8000/v1` inside the Compose network. PostgreSQL is included for the deployment exercise, while the current ticket tool still uses the local SQLite lab database.

The supplied public URL is `https://aitss.xor01.com/`, but it returned HTTP 404 during the 2026-09-20 verification. It is therefore a deployment reference, not verified demo evidence.
