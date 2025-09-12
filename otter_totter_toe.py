# Imports
from IPython.display import display
import numpy as np
import random
from abc import ABC, abstractmethod
from collections import defaultdict
import plotly.express as px
import ipywidgets as widgets
import torch
import torch.nn as nn
import torch.optim as optim

# Set dark theme for all Plotly charts
px.defaults.template = 'plotly_dark'

# Tic-Tac-Toe environment definition
class TicTacToeEnv:
    """
    Tic-Tac-Toe environment:
    - Board: 3x3 numpy array, values {1: X, -1: O, 0: empty}
    - current_player: 1 (X) or -1 (O)
    Methods:
      reset(): clear board, X starts
      available_actions(): list of empty cells
      step(action): place mark, switch player, return (state, reward, done, info)
      check_winner(): detect 3-in-row
    """
    def __init__(self, size =3):
        self.board = np.zeros((3, 3), dtype=int)
        self.current_player = 1

    def reset(self):
        """Reset board and set starting player to X (1)."""
        self.board.fill(0)
        self.current_player = 1
        return self.board.copy()

    def available_actions(self):
        """Return list of (row, col) for empty cells."""
        # TODO: implement
        return [(i, j) for i in range(3) for j in range(3) if self.board[i, j] == 0]

    def step(self, action):
        """
        Apply move for current_player.
        Returns:
          state: new board copy
          reward: +1 current player win, 0 otherwise
          done: True if win or draw
          info: empty dict
        """
        i, j = action
        if self.board[i, j] != 0:
            raise ValueError(f"Attempted invalid move at {action}")

        # Place mark
        self.board[i, j] = self.current_player
        winner = self.check_winner()
        done = (winner != 0) or (len(self.available_actions()) == 0)

        # Assign reward
        if winner == self.current_player:
            reward = 1
        else:
            reward = 0

        # Switch turn
        self.current_player *= -1
        return self.board.copy(), reward, done, {}

    def check_winner(self):

        """
        Check all rows, cols, diags for a winner.
        If there is a winner, return their associated value (1 or -1).
        If there is no winner, return 0.
        """

        # TODO: implement
        for i in range(3):
            if abs(np.sum(self.board[i, :])) == 3:
                return np.sign(np.sum(self.board[i, :]))

            if abs(np.sum(self.board[:, i])) == 3:
                return np.sign(np.sum(self.board[:, i]))


        diag1 = np.sum([self.board[i, i] for i in range(3)])
        if abs(diag1) == 3:
            return np.sign(diag1)

        diag2 = np.sum([self.board[i, 2 - i] for i in range(3)])
        if abs(diag2) == 3:
            return np.sign(diag2)

        return 0
    
# GUI definition
def start_gui(agent, env):
        """
        Launch interactive Tic-Tac-Toe board:
        - Human plays X, agent plays O
        - Dynamic status label
        - Reset capability
        """
        # Create status display
        status = widgets.HTML(value="<h3>Player X's turn</h3>")
        # Create reset button
        btn_reset = widgets.Button(description='Reset Game', button_style='info')

        # Create 3x3 grid of buttons
        buttons = [[widgets.Button(layout=widgets.Layout(width='80px', height='80px'),
                                    style={'button_color': 'lightgray'}) for _ in range(3)]
                for _ in range(3)]
        grid = widgets.GridBox([btn for row in buttons for btn in row],
            layout=widgets.Layout(grid_template_columns='repeat(3,80px)', grid_gap='3px'))

        # Display UI
        display(widgets.VBox([status, btn_reset, grid]))

        def reset_game(_=None):
            """Clear board UI and reset environment."""
            env.reset()
            status.value = "<h3>Player X's turn</h3>"
            for i in range(3):
                for j in range(3):
                    btn = buttons[i][j]
                    btn.description = ' '
                    btn.disabled = False
                    btn.style.button_color = 'lightgray'

        btn_reset.on_click(reset_game)

        def update_status():
            """Update status label based on game state."""
            w = env.check_winner()
            if w == 1:
                status.value = "<h3 style='color:green;'>X wins!</h3>"
            elif w == -1:
                status.value = "<h3 style='color:red;'>O wins!</h3>"
            elif not env.available_actions():
                status.value = "<h3 style='color:orange;'>Draw!</h3>"
            else:
                turn = 'X' if env.current_player == 1 else 'O'
                status.value = f"<h3>Player {turn}'s turn</h3>"

        def place_X(i, j):
            assert(env.board[i,j] == 1)
            buttons[i][j].description = 'X'
            buttons[i][j].style.button_color = '#00bfff'
            buttons[i][j].disabled = True

        def place_O(i, j):
            assert(env.board[i,j] == -1)
            buttons[i][j].description = 'O'
            buttons[i][j].style.button_color = '#ff6347'
            buttons[i][j].disabled = True

        def on_click(i, j):
            def handler(_):
                # Human move
                if env.board[i, j] != 0:
                    # Invalid move, do nothing
                    return
                _, _, done, _ = env.step((i, j))
                place_X(i, j)
                update_status()
                if done:
                    return

                # Agent move
                action = agent.policy(env.board.copy(), env)
                _, _, done, _ = env.step(action)
                ai, aj = action
                place_O(ai, aj)
                update_status()

            return handler

        # Attach handlers
        for i in range(3):
            for j in range(3):
                buttons[i][j].on_click(on_click(i, j))

class TicTacToeAgent(ABC):
    @abstractmethod
    def policy(self, state, env):
        """
        Given the state and the environment, return the action
        corresponding to the agent's learned best policy.
        Used when evaluating the agent after training.
        """
        pass

    @abstractmethod
    def training_policy(self, state, env):
        """
        Given the state and the environment, return the action
        corresponding to the agent's policy.
        Used when training the agent.
        """
        pass

    @abstractmethod
    def learn(self, *args, **kwargs):
        
        pass

# Random Agent Baseline
class RandomAgent(TicTacToeAgent):
    """Baseline: selects moves at random."""
    def policy(self, state, env):
        # TODO: implement
        return random.choice(env.available_actions())

    def training_policy(self, state, env):
        # TODO: implement
        return random.choice(env.available_actions())

    def learn(self, *args, **kwargs):
        # TODO: implement
        pass

# Play against baseline random agent
env = TicTacToeEnv()
print("### Playing against RandomAgent ###")
start_gui(RandomAgent(), env)