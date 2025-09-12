# save this as app.py
from flask import Flask, request, jsonify
from flask_cors import CORS
import random
import numpy as np

# Use your existing environment and agent
from otter_totter_toe.py import TicTacToeEnv, RandomAgent  

app = Flask(__name__)
CORS(app)  # allow HTML/JS to connect

env = TicTacToeEnv()
agent = RandomAgent()

@app.route("/reset", methods=["POST"])
def reset():
    env.reset()
    return jsonify({"board": env.board.tolist(), "current_player": env.current_player})

@app.route("/move", methods=["POST"])
def move():
    data = request.json
    i, j = data["row"], data["col"]

    # Human move
    state, reward, done, _ = env.step((i, j))
    result = {"board": state.tolist(), "done": done, "winner": env.check_winner()}

    if not done:
        # AI move
        action = agent.policy(state, env)
        state, _, done, _ = env.step(action)
        result["board"] = state.tolist()
        result["done"] = done
        result["winner"] = env.check_winner()
        result["ai_move"] = action

    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True)