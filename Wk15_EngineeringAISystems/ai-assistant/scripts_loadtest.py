"""Concurrency / throughput benchmark for the deployed API.

  uv run python scripts_loadtest.py --endpoint route --n 500 --concurrency 32
  uv run python scripts_loadtest.py --endpoint chat  --n 20  --concurrency 4
Writes docs/loadtest_<endpoint>.json (p50/p95 latency, throughput, error rate).
"""
import argparse
import asyncio
import json
import statistics
import time
from pathlib import Path

import httpx

MSGS = ["I want a refund for my last order", "how do I change my address", "delete my account please",
        "charged twice on my card", "where is my invoice", "talk to a human", "cancel order 123"]
QS = ["What code representation does AI4VA use?", "What recall does VUDENC report?",
      "How does LineVD detect statement-level vulnerabilities?"]


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--api", default="http://localhost:8080")
    ap.add_argument("--endpoint", choices=["route", "chat"], default="route")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--concurrency", type=int, default=32)
    a = ap.parse_args()
    sem, lat, errs = asyncio.Semaphore(a.concurrency), [], 0
    async with httpx.AsyncClient(timeout=120, headers={"x-api-key": f"loadtest-{time.time()}"}) as c:
        async def one(i):
            nonlocal errs
            async with sem:
                payload = {"message": MSGS[i % len(MSGS)]} if a.endpoint == "route" else {"question": QS[i % len(QS)]}
                t0 = time.perf_counter()
                r = await c.post(f"{a.api}/{a.endpoint}", json=payload)
                lat.append((time.perf_counter() - t0) * 1000)
                errs += r.status_code != 200
        t0 = time.perf_counter()
        await asyncio.gather(*(one(i) for i in range(a.n)))
        wall = time.perf_counter() - t0
    lat.sort()
    out = {"endpoint": a.endpoint, "n": a.n, "concurrency": a.concurrency, "throughput_rps": round(a.n / wall, 1),
           "p50_ms": round(statistics.median(lat), 1), "p95_ms": round(lat[int(0.95 * len(lat)) - 1], 1),
           "error_rate": errs / a.n}
    Path("docs").mkdir(exist_ok=True)
    Path(f"docs/loadtest_{a.endpoint}.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
