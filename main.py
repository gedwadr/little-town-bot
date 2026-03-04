import json
import uvicorn
from fastapi import FastAPI, Request

app = FastAPI()


@app.post("/move")
async def move(request: Request):
    body = await request.json()
    print(json.dumps(body, indent=2))
    return {"move": "placeWorker", "args": []}


def main():
    uvicorn.run("main:app", host="0.0.0.0", port=9000, reload=True)


if __name__ == "__main__":
    main()
