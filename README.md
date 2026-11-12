# Chatbot — Student Doubt Resolution

Chatbot is an AI-assisted question-resolution system for an education product. It combines conversational state, retrieval-augmented generation, relevance checks, abuse and gibberish detection, structured LLM classification, confidence thresholds, and conservative fallbacks.

This repository is a focused portfolio slice of a larger Django product. It intentionally excludes production settings, credentials, databases, customer data, and unrelated application code.

## Why this project

Students do not always describe a problem using the category that best matches it. The chatbot therefore treats question resolution as a routing problem:

1. Load the question and conversation context.
2. Classify the message for abuse, gibberish, and relevance.
3. Retrieve similar approved responses when confidence is high enough.
4. Ask an LLM to conservatively reclassify ambiguous “Other” doubts.
5. Route the student to a targeted explanation flow, or safely fall back to “Other”.
6. Log the decision, confidence, fallback reason, and latency for evaluation.

## Architecture

```text
Student message
      |
      v
Conversation context --> Abuse / gibberish / relevance detectors
      |                              |
      |                              +--> safe rejection or explanation
      v
RAG retrieval + confidence threshold
      |
      +--> approved answer when confidence is high
      |
      +--> LLM reclassification for ambiguous “Other” doubts
                    |
                    +--> targeted handler or conservative fallback
```

See [architecture.md](architecture.md) for design decisions and tradeoffs.

## Technical highlights

- Provider abstractions for LLM and embedding services
- Dependency-injected classifiers and repositories
- Typed result objects with explicit reason codes
- JSON response validation and timeout handling
- Confidence thresholds to reduce unsafe or low-quality routing
- RAG relevance and retrieval strategies
- Test fixtures for routing and reclassification behavior

## Running the tests

The extracted production slice depends on the original Django application context. The tests and fixtures are included as a reviewable reference. To run the complete implementation, provide the host application's Django settings and dependencies.

```bash
python -m pytest tests -q
```

## Configuration

Copy `.env.example` to `.env` and provide credentials only in your local environment. Never commit secrets.

## Portfolio note

This project was built as part of LogiQids, an education platform supporting online exams, worksheets, gamification, and student support. The repository is intentionally narrowed to the chatbot subsystem so the architecture and engineering decisions are easy to review.
