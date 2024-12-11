import torch
import numpy as np
from transformer_model import GameplayTransformer

class RLAgent:
    def __init__(self, model_path, input_size, num_heads, hidden_size, num_layers, output_size, lr=0.0001, gamma=0.99):
        """
        Reinforcement Learning agent using Transformer model for gameplay.
        Args:
            model_path (str): Path to the pre-trained Transformer model.
        """
        self.model = GameplayTransformer(input_size, num_heads, hidden_size, num_layers, output_size)
        self.model.load_state_dict(torch.load(model_path))
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.gamma = gamma
        self.criterion = torch.nn.MSELoss()

    def choose_action(self, state):
        with torch.no_grad():
            state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            probabilities = self.model(state)
            action = torch.argmax(probabilities).item()
        return action

    def train(self, states, actions, rewards):
        """
        Train the agent using collected experience.
        Args:
            states (list): List of observed states.
            actions (list): List of actions taken.
            rewards (list): List of rewards received.
        """
        discounted_rewards = self.discount_rewards(rewards)
        for state, action, reward in zip(states, actions, discounted_rewards):
            state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            action = torch.tensor([action], dtype=torch.long)
            reward = torch.tensor([reward], dtype=torch.float32)

            self.optimizer.zero_grad()
            output = self.model(state)
            log_prob = torch.log(output[0, action])
            loss = -log_prob * reward
            loss.backward()
            self.optimizer.step()

    def discount_rewards(self, rewards):
        """
        Compute discounted rewards.
        Args:
            rewards (list): List of rewards.
        Returns:
            list: Discounted rewards.
        """
        discounted = []
        cumulative = 0
        for reward in reversed(rewards):
            cumulative = reward + self.gamma * cumulative
            discounted.insert(0, cumulative)
        return discounted

if __name__ == "__main__":
    agent = RLAgent(
        model_path="models/transformer/transformer_model_finetuned.pth",
        input_size=128, num_heads=4, hidden_size=64, num_layers=2, output_size=10
    )
    # Example usage:
    states = [np.random.rand(128) for _ in range(10)]
    actions = [np.random.randint(0, 10) for _ in range(10)]
    rewards = [np.random.rand() for _ in range(10)]
    agent.train(states, actions, rewards)
