# Deep Q-Learning Snake Game

This project is an implementation of the classic Snake game using Deep Q-Learning. The Snake game is a simple yet challenging game where a snake moves around the screen, eating food to grow longer while avoiding collisions with the walls and itself.

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Usage](#usage)
- [Contributing](#Contributing)

## Introduction

- **Reinforcement Learning (RL):**
Reinforcement Learning is a field of machine learning where an agent learns to make decisions by interacting with an environment to maximize cumulative rewards.

- **Q-Learning:**
Q-Learning is a RL algorithm that learns the expected rewards (Q-values) for taking actions in states. It uses the Bellman equation to iteratively update Q-values based on rewards and transitions.

- **Bellman Equation:**
The Bellman equation describes how Q-values are updated. It balances the immediate reward (r) and the expected future rewards (max(Q(s', a'))) with a learning rate (α) and a discount factor (γ).
```
Q(s, a) = (1 - α) * Q(s, a) + α * [r + γ * max(Q(s', a'))]
```

- **Deep Q-Learning (DQN):**
Deep Q-Learning extends Q-Learning by using deep neural networks to estimate Q-values. It's ideal for handling high-dimensional inputs, such as images, and employs techniques like experience replay and target networks for stable training.

- In the context of the Snake game, DQN helps the agent learn optimal strategies for maximizing its score by eating food and avoiding collisions.

## Features

- Snake game with a graphical user interface.
- Q-Learning agent trained to play the game.
- Save and load trained models for future gameplay.
- Adjustable hyperparameters for training.

## Usage

Follow these steps to run the Snake game and the Deep Q Learning agent:

1. **Clone or Fork the Project**

2. **Create a virtual environment**
    
    ```python
    # For Windows and Linux
    python3 -m venv <venv name>

    # For macOS
    python3 -m venv <venv name>

3. **Install Dependencies:**

   Before running the game, ensure you have the necessary dependencies installed:

   ```python
   # For Windows and Linux
   pip install -r requirements.txt
   
   # For macOS
   pip3 install -r requirements.txt

4. **To play the snake game:**
    
    ```python
   # For Windows and Linux
   python snake_game.py
   
   # For macOS
   python3 snake_game.py

5. **To run the Deep Q Learning Agent:**

    ```python
    cd Deep_Q_Learning

   # For Windows and Linux
   python agent.py
   
   # For macOS
   python3 agent.py

6. **To Run a Custom Model on the Agent:**

   If you have a pretrained model and would like to run it with the agent, follow these steps:

   - Ensure your model is saved in a compatible format (e.g., PyTorch `.pth` file).

   - Open the `pretrained_agent.py` script in the "Deep Q Learning" directory.

   - Update the `model_path` variable in the script to point to the location of your pretrained model file.

   - Run the `pretrained_agent.py` script using the following command:

     ```python
     cd Deep_Q_Learning
     python pretrained_agent.py
     ```

   This will load your custom model and run the agent with it, allowing you to observe how your model performs in the Snake game.


## Contributing

I appreciate feedback and suggestions. If you have any ideas for improvements or spot any issues, please create an issue or submit a pull request.
