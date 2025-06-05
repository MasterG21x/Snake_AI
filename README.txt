🐍 Snake with Deep Q-Network (DQN) Agent
========================================

This project implements the classic Snake game in Python, featuring an agent that learns using a Deep Q-Network (DQN). The game is built using Pygame for rendering, and TensorFlow powers the DQN.

──────────────────────────────
🗂️  Project Structure
──────────────────────────────
- snake.py     — Snake game logic: snake movement, collisions
- food.py      — Food generation on the board
- settings.py  — Game settings (window size, colors)
- app.py       — Main Board class integrating the game and AI
- model.py     — Defines the QNetwork and the QTrainer class for training
- agent.py     — DQN agent implementation
- main.py      — Training and running the agent

──────────────────────────────
🕹️  How to Run the Game Manually
──────────────────────────────
1️⃣  Install dependencies:
    pip install pygame tensorflow numpy

2️⃣  Run the app manually:
    python app.py

   (In app.py, set `headless=False` to enable the graphical display).

──────────────────────────────
🤖 How to Train the DQN Agent
──────────────────────────────
1️⃣  Run main.py:
    python main.py

2️⃣  The agent will play episodes of the game and learn from experience.
    Every 50 episodes, rendering can be toggled on or off.

──────────────────────────────
⚙️  Key Features
──────────────────────────────
✅  DQN agent with replay memory
✅  Epsilon-greedy exploration strategy
✅  Neural network (QNetwork): 2 hidden layers (256 neurons each)
✅  Snake game environment in Pygame with collision detection and food

──────────────────────────────
📌  Notes
──────────────────────────────
- By default, the agent is trained in headless mode (no graphics) for faster training.

