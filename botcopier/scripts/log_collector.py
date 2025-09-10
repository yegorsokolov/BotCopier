#!/usr/bin/env python3
"""Simple OTLP HTTP log collector."""
import argparse
import json
from aiohttp import web

async def handle(request):
    payload = await request.read()
    try:
        data = json.loads(payload.decode())
        print(json.dumps(data))
    except Exception:
        print(payload)
    return web.Response(text="")

def main() -> None:
    parser = argparse.ArgumentParser(description="OTLP log collector")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=4318)
    args = parser.parse_args()
    app = web.Application()
    app.add_routes([web.post("/v1/logs", handle)])
    web.run_app(app, host=args.host, port=args.port)

if __name__ == "__main__":
    main()
