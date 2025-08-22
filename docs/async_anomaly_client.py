"""Example asynchronous client for the anomaly service.

The script submits a payload and then polls for the result without
blocking the rest of the application.  It demonstrates how external
workers can be used to offload heavy processing.
"""

import asyncio
import aiohttp
import uuid

ANOMALY_URL = "http://127.0.0.1:8000/anomaly"


async def enqueue(session, payload):
    job_id = uuid.uuid4().hex
    await session.post(ANOMALY_URL, json={"id": job_id, "payload": payload})
    return job_id


async def poll(session, job_id):
    url = f"{ANOMALY_URL}?id={job_id}"
    while True:
        async with session.get(url) as resp:
            if resp.status == 200:
                txt = await resp.text()
                return float(txt)
        await asyncio.sleep(0.5)


async def main():
    async with aiohttp.ClientSession() as session:
        job = await enqueue(session, [1, 2, 3, 4, 5, 6])
        print("enqueued", job)
        score = await poll(session, job)
        print("anomaly score", score)


if __name__ == "__main__":
    asyncio.run(main())
