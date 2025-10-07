
#!/usr/bin/env bash
uvicorn src.infer_service:app --host 0.0.0.0 --port 8080 --loop asyncio --reload
