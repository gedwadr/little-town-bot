import random
import time
import requests

URL = "http://localhost:8000/api/bot-match"
INTERVAL_SECONDS = 75


def trigger_match():
    payload = {
        "numPlayers": 4,
        "boardSide": random.choice(["A", "B"]),
        "randomizeTurnOrder": True,
        "bots": [
            {"playerID": "0", "serviceUrl": "http://localhost:9002", "botName": "ResNet-SFT"},
            {"playerID": "1", "serviceUrl": "http://localhost:9001", "botName": "ResNet-RL"},
            {"playerID": "2", "serviceUrl": "http://localhost:9002", "botName": "ResNet-SFT"},
            {"playerID": "3"},
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
