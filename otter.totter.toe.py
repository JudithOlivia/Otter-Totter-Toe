# Imports
from readline import redisplay
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
        redisplay(widgets.VBox([status, btn_reset, grid]))

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

# EXTENSION
# Neural Network for Q-function
class QNetwork(nn.Module):
    def __init__(self, input_dim=9, hidden_dim=128, output_dim=9):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


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
        """
        Carry out any policy updates/other internal state updates.
        Does not need to return anything.
        """
        pass

class NeuralQAgent(TicTacToeAgent):
    """Neural Network Q-Learning Agent for Tic-Tac-Toe"""
    def __init__(self, alpha=0.001, gamma=0.9, epsilon=0.3, decay=0.9995, min_eps=0.05):
        self.gamma = gamma
        self.epsilon = epsilon
        self.decay = decay
        self.min_eps = min_eps

        self.model = QNetwork()
        self.optimizer = optim.Adam(self.model.parameters(), lr=alpha)
        self.loss_fn = nn.MSELoss()

    def state_to_tensor(self, state):
        """Flatten board into torch tensor"""
        return torch.tensor(state.flatten(), dtype=torch.float32).unsqueeze(0)

    def policy(self, state, env):
        """Greedy policy"""
        actions = env.available_actions()
        state_tensor = self.state_to_tensor(state)
        q_values = self.model(state_tensor).detach().numpy().flatten()

      # Mask illegal moves by assigning very negative value
        mask = np.full(9, -1e9)
        for (i, j) in actions:
            mask[i * 3 + j] = q_values[i * 3 + j]
        best_idx = np.argmax(mask)
        return (best_idx // 3, best_idx % 3)

    def training_policy(self, state, env):
        """ε-greedy policy"""
        if random.random() < self.epsilon:
            return random.choice(env.available_actions())
        return self.policy(state, env)

    def learn(self, state, action, reward, next_state, env):
        state_tensor = self.state_to_tensor(state)
        next_state_tensor = self.state_to_tensor(next_state)

        # Predict current Q
        q_values = self.model(state_tensor)
        q_value = q_values[0, action[0]*3 + action[1]]

        # Compute target
        next_q_values = self.model(next_state_tensor).detach().numpy().flatten()
        if env.available_actions():
            future = max(next_q_values[a[0]*3 + a[1]] for a in env.available_actions())
        else:
            future = 0
        target = reward + self.gamma * future

        # Backprop
        loss = self.loss_fn(q_value, torch.tensor(target, dtype=torch.float32))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Decay epsilon
        self.epsilon = max(self.min_eps, self.epsilon * self.decay)
        self.epsilon = max(self.min_eps, self.epsilon * self.decay)