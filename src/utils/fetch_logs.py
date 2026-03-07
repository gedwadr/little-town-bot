import asyncio
import json
from pathlib import Path
import os

import httpx

BASE_URL = os.environ["BASE_URL"]
OUTPUT_DIR = Path("./logs/matchLogs")
MAX_PAGES = 100
CONCURRENCY = 10  # simultaneous match-detail requests


async def fetch_all_match_ids(client: httpx.AsyncClient) -> list[str]:
    match_ids: list[str] = []
    for page in range(1, MAX_PAGES + 1):
        resp = await client.get(f"{BASE_URL}/matches", params={"page": page})
        resp.raise_for_status()
        data = resp.json()
        matches = data.get("matches", [])
        if not matches:
            print(f"Page {page}: no more matches, stopping.")
            break
        for m in matches:
            match_ids.append(m["match_id"])
        print(f"Page {page}: found {len(matches)} matches (total so far: {len(match_ids)})")
        if page >= data.get("total", 0) // data.get("limit", 20) + 1:
            break
    return match_ids


async def fetch_and_save_match(
    client: httpx.AsyncClient,
    sem: asyncio.Semaphore,
    match_id: str,
) -> None:
    out_path = OUTPUT_DIR / f"{match_id}.txt"
    if out_path.exists():
        print(f"  [{match_id}] already saved, skipping.")
        return

    async with sem:
        resp = await client.get(f"{BASE_URL}/matches/{match_id}")
        resp.raise_for_status()
        data = resp.json()

    out_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    print(f"  [{match_id}] saved.")


async def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    async with httpx.AsyncClient(timeout=30) as client:
        print("=== Fetching match list ===")
        match_ids = await fetch_all_match_ids(client)
        print(f"\nTotal matches to download: {len(match_ids)}\n")

        print("=== Downloading match details ===")
        sem = asyncio.Semaphore(CONCURRENCY)
        tasks = [fetch_and_save_match(client, sem, mid) for mid in match_ids]
        await asyncio.gather(*tasks)

    print("\nDone.")


if __name__ == "__main__":
    asyncio.run(main())
