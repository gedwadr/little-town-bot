import random
import time
import requests

URL = "http://localhost:8000/api/bot-match"
INTERVAL_SECONDS = 90


def trigger_match():
    payload = {
        "numPlayers": 2,
        "boardSide": random.choice(["A", "B"]),
        "randomizeTurnOrder": True,
        "bots": [
            {"playerID": "0"},
            {"playerID": "1", "serviceUrl": "http://localhost:9000", "botName": "QwenSFT-RL"},
        ],
    }
    try:
        res = requests.post(URL, json=payload, timeout=10)
        data = res.json()
        print(f"[cron] match started: {data}")
    except Exception as e:
        print(f"[cron] error: {e}")


if __name__ == "__main__":
    print(f"[cron] starting — firing every {INTERVAL_SECONDS}s")
    while True:
        trigger_match()
        time.sleep(INTERVAL_SECONDS)
