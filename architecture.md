# Architecture notes

## Routing before generation

The system separates classification from answer generation. This makes the behavior easier to test and lets the product route known cases to targeted instructions instead of asking a general-purpose model to solve every request.

## Conservative ambiguity handling

The “Other” classifier only changes the route when the model returns valid JSON, a recognized category, a valid confidence value, and confidence above the configured threshold. API failures, malformed output, invalid categories, and low confidence all fall back to the original category.

## Retrieval confidence

The RAG layer checks relevance and retrieves candidate responses through injected interfaces. A response is returned only when retrieval confidence clears the configured threshold; otherwise the caller can continue through a safer fallback path.

## Observability

Classification results carry reason codes, handler identifiers, fallback reasons, and latency. This makes model behavior measurable rather than treating the LLM as an opaque dependency.

## Production improvements I would make next

- Add a versioned evaluation dashboard for precision, fallback rate, and latency.
- Add per-user rate limits and circuit breakers around provider calls.
- Move approved response retrieval to a managed vector index when scale requires it.
- Add contract tests for every provider implementation.
