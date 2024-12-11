import torch
from torch.utils.data import DataLoader
from nn_model import GameplayNN
from nn_training import GameplayDataset

def finetune_nn_model(model_path, dataset_path, output_model_path, epochs=5, batch_size=16, lr=0.0001):
    """
    Fine-tune the Neural Network for real-time performance.
    Args:
        model_path (str): Path to the pre-trained model.
        dataset_path (str): Path to the fine-tuning dataset.
        output_model_path (str): Path to save the fine-tuned model.
    """
    dataset = GameplayDataset(dataset_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = GameplayNN(input_size=128, hidden_size=64, output_size=10)
    model.load_state_dict(torch.load(model_path))

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for features, actions in dataloader:
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, actions)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Fine-tuning Epoch {epoch+1}/{epochs}, Loss: {total_loss:.4f}")

    torch.save(model.state_dict(), output_model_path)
    print(f"Fine-tuned model saved to {output_model_path}")

if __name__ == "__main__":
    finetune_nn_model(
        model_path="models/neural_network/nn_model.pth",
        dataset_path="data/processed/nn_dataset.csv",
        output_model_path="models/neural_network/nn_model_finetuned.pth"
    )
