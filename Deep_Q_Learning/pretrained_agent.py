import torch
from agent import Agent
from snakeAI import SnakeAI, Direction, Point
from model import Linear_QNet
import numpy as np


class PretrainedAgent:
    def __init__(self, model_path):
        # Create an instance of the SnakeAI game environment
        self.game = SnakeAI()

        # Load the pre-trained model
        # Make sure to use the same architecture as during training
        self.model = Linear_QNet(11, 256, 3)
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()  # Set the model in evaluation mode (no gradient calculations)

        # Create an instance of the Agent and set the loaded model
        self.agent = Agent()
        self.agent.model = self.model

    def run_game(self):
        done = False

        while not done:
            # Get the current game state
            state = self.agent.get_state(self.game)

            # Use the agent's model to select the action
            state_tensor = torch.tensor(state, dtype=torch.float)
            with torch.no_grad():  # Disable gradient tracking during inference
                prediction = self.agent.model(state_tensor)
                move = torch.argmax(prediction).item()

            # Convert the action index to a one-hot vector
            final_move = [0, 0, 0]
            final_move[move] = 1

            # Take a step in the game environment
            reward, done, score = self.game.play_step(final_move)

        print(f"Game Over! Score: {self.game.score}")


if __name__ == "__main__":
    model_path = "model/model.pth"
    game_runner = PretrainedAgent(model_path)
    game_runner.run_game()
