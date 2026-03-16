# import json
# import uvicorn
# from fastapi import FastAPI, Request, HTTPException
#
# from src.games.action_validator import ActionValidator
# from src.games.botV1 import BotV1
# from src.games.game_parser import GameParser
#
# app = FastAPI()
#
#
# @app.post("/move")
# async def move(request: Request):
#     body = await request.json()
#     player_id = body.get("playerID", "999")
#     if player_id == "9999":
#         raise HTTPException(status_code=400, detail="Player ID is invalid")
#     game_parser = GameParser.from_live_data(body)
#     prompt = game_parser.build_prompt(player_id=player_id)
#     with open("live_prompt.json", "w") as f:
#         json.dump(body, f)
#     bot = BotV1(
#         model_path="./models/qwen2.5-1.5b-instruct",
#         adapter_path="./checkpoints/boardgame-v1/checkpoint-1331"
#     )
#     validator = ActionValidator(game_parser)
#
#     moves = body.get("moves", [])
#     raw = bot.get_actions(prompt["system"], prompt["user"])
#
#     chosen = validator.to_majapahit_move(raw, player_id=player_id, moves=moves)
#     if chosen is None:
#         if moves:
#             chosen = moves[0]
#         else:
#             empties = game_parser.board.empty_grass_cells()
#             r, c = empties[0] if empties else (0, 0)
#             chosen = {"move": "placeWorker", "args": [r, c, []]}
#
#     print(moves)
#     print("=====" * 30)
#     print("Chosen move:", chosen)
#     print("=====" * 30)
#
#     return chosen
#
#
# def main():
#     uvicorn.run("src.main:app", host="0.0.0.0", port=9000, reload=True)
#
#
# if __name__ == "__main__":
#     main()
