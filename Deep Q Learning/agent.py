import torch
import random
import numpy as np
from collections import deque
from snakeAI import SnakeAI, Direction, Point
from model import Linear_QNet, QTrainer
from helper import plot

# max memory for the deque; we can adjust to our liking
MAX_MEMORY = 100_000

# batch size for our model input
BATCH_SIZE = 1000
LR = 0.001


class Agent:

    def __init__(self):
        # Initialize agent's attributes
        self.n_games = 0  # Counter for the number of games played
        self.gamma = 0.9  # Discount rate for future rewards
        self.memory = deque(maxlen=MAX_MEMORY)  # Replay memory for experiences
        # Q-value estimation neural network
        self.model = Linear_QNet(11, 256, 3)
        # Trainer for the neural network
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)

    def get_state(self, snakeAI):
        # Calculate and return the game state as a feature vector
        head = snakeAI.snake[0]

        point_l = Point(head.x - 20, head.y)
        point_r = Point(head.x + 20, head.y)
        point_u = Point(head.x, head.y - 20)
        point_d = Point(head.x, head.y + 20)

        dir_l = snakeAI.direction == Direction.LEFT
        dir_r = snakeAI.direction == Direction.RIGHT
        dir_u = snakeAI.direction == Direction.UP
        dir_d = snakeAI.direction == Direction.DOWN

        state = [
            # Danger straight
            (dir_l and snakeAI.is_collision(point_l)) or
            (dir_r and snakeAI.is_collision(point_r)) or
            (dir_u and snakeAI.is_collision(point_u)) or
            (dir_d and snakeAI.is_collision(point_d)),

            # Danger right
            (dir_l and snakeAI.is_collision(point_u)) or
            (dir_r and snakeAI.is_collision(point_d)) or
            (dir_u and snakeAI.is_collision(point_r)) or
            (dir_d and snakeAI.is_collision(point_l)),

            # Danger left
            (dir_l and snakeAI.is_collision(point_d)) or
            (dir_r and snakeAI.is_collision(point_u)) or
            (dir_u and snakeAI.is_collision(point_l)) or
            (dir_d and snakeAI.is_collision(point_r)),

            # move direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,

            #
            snakeAI.food.x < snakeAI.head.x,  # food left
            snakeAI.food.x > snakeAI.head.x,  # food right
            snakeAI.food.y < snakeAI.head.y,  # food up
            snakeAI.food.y > snakeAI.head.y  # food down

        ]

        return np.array(state, dtype=int)

    def remember(self, state, action, reward, next_state, done):
        # Store an experience tuple in the replay memory
        self.memory.append((state, action, reward, next_state, done))

    def train_long_memory(self):
        # Train the model using experiences from replay memory
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(
                self.memory, BATCH_SIZE)  # list of tuples
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, done):
        # Train the model with a single experience
        self.trainer.train(state, action, reward, next_state, done)

    def get_action(self, state):
        # Determine the action to take based on the current state
        # random moves; exploration vs exploitation
        # Exploration rate (controls randomness)
        self.epsilon = 80 - self.n_games
        final_move = [0, 0, 0]
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1

        return final_move


def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent = Agent()
    game = SnakeAI()

    while True:
        state_old = agent.get_state(game)
        final_move = agent.get_action(state_old)
        reward, done, score = game.play_step(final_move)
        state_new = agent.get_state(game)
        agent.train_short_memory(
            state_old, final_move, reward, state_new, done)

        agent.remember(state_old, final_move, reward, state_new, done)

        if done:
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()
            if score > record:
                record = score
                agent.model.save()

            print(f'Game {agent.n_games} Score {score} Record {record}')

            plot_scores.append(score)
            total_score += score
            mean_score = total_score/agent.n_games
            plot_mean_scores.append(mean_score)
            plot(plot_scores, plot_mean_scores)


if __name__ == '__main__':
    train()
