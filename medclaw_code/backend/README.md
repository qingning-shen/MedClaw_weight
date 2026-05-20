# backend

FastAPI server that routes frontend chat requests to the smolagents agent.

## Requirements

```bash
pip install -r requirements.txt
# Also install agent dependencies:
pip install -r ../agent/requirements.txt
```

## Run

```bash
# With the real model server running
uvicorn main:app --host 0.0.0.0 --port 5000

# Local testing without vLLM (mock responses)
MEDCLAW_TEST_MODE=1 uvicorn main:app --host 0.0.0.0 --port 5000 --reload
```

## API

### `POST /chat`

**Request**
```json
{"message": "Find papers about BRCA1 breast cancer"}
```

**Response**
```json
{
  "answer": "I found the following papers...",
  "steps": [
    {
      "tool": "pubmed_search",
      "input": {"query": "BRCA1 breast cancer", "max_results": 5},
      "output": "1. Paper title... | Authors | Journal (2024)"
    }
  ]
}
```

### `GET /health`

Returns `{"status": "ok", "test_mode": false}`.

## Environment variables

Same as the agent — see `../agent/README.md` for the full list.
