import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os

# Define a neural network class for the Q-learning model.


class Linear_QNet(nn.Module):

    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        # Define two linear layers for the neural network.
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # Implement the forward pass of the neural network with ReLU activation.
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x

    def save(self, file_name='model.pth'):
        # Save the model's state dictionary to a file with the given name.
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)

# Define a class for training the Q-learning model.


class QTrainer:

    def __init__(self, model, lr, gamma):
        self.model = model
        self.lr = lr
        self.gamma = gamma
        # Initialize an Adam optimizer for training the model.
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        # Define the mean squared error (MSE) loss criterion.
        self.criterion = nn.MSELoss()

    def train(self, state, action, reward, next_state, done):
        # Convert input data to PyTorch tensors.
        state = torch.tensor(state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.float)
        reward = torch.tensor(reward, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)

        if len(state.shape) == 1:
            # If the input tensors are 1D, expand them to have a batch dimension.
            state = torch.unsqueeze(state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            next_state = torch.unsqueeze(next_state, 0)
            done = (done, )

        # Predicted Q-values with the current state.
        pred = self.model(state)

        # Calculate the target Q-values for the given state-action pairs.
        target = pred.clone()
        for index in range(len(done)):
            Q_new = reward[index]
            if not done[index]:
                # Update Q-value using the Bellman equation.
                Q_new = reward[index] + self.gamma * \
                    torch.max(self.model(next_state[index]))

            target[index][torch.argmax(action).item()] = Q_new

        # Zero the gradients, compute the loss, and perform backpropagation.
        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()

        # Update the model's weights using the optimizer.
        self.optimizer.step()
