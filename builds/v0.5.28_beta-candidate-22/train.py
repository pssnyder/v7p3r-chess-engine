from chess_core import *
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import yaml
import numpy as np

torch.backends.cudnn.benchmark = True  # Enable CuDNN auto-tuner

# Load configuration
with open("config.yaml") as f:
    config = yaml.safe_load(f)

# Calculate move weights based on frequency
def calculate_move_weights(moves_list, num_classes):
    counts = np.bincount(moves_list, minlength=num_classes)
    weights = 1.0 / (np.sqrt(counts) + 1e-6)  # Inverse sqrt frequency
    return torch.tensor(weights / weights.max(), dtype=torch.float32)

# Initialize move-to-index mapping
def create_move_vocab(dataset):
    return {move: idx for idx, move in enumerate(np.unique(dataset.moves))}

def train_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load data and create vocabulary
    dataset = ChessDataset("games.pgn", "v7p3r")
    move_to_index = create_move_vocab(dataset)
    num_classes = len(move_to_index)
    
    # Save the move vocabulary to a file
    import pickle
    with open("move_vocab.pkl", "wb") as f:
        pickle.dump(move_to_index, f)
    
    # Convert string moves to indices
    move_indices = [move_to_index[move] for move in dataset.moves]
    dataset.moves = move_indices
    
    dataloader = DataLoader(
        dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    model = ChessAI(num_classes).to(device)
    criterion = nn.CrossEntropyLoss(
        weight=calculate_move_weights(np.array(dataset.moves), num_classes)
    )
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )
    
    for epoch in range(config['training']['epochs']):
        model.train()
        for positions, moves in dataloader:
            positions = positions.to(device, non_blocking=True)
            moves = moves.to(device, non_blocking=True)
            
            outputs = model(positions)
            loss = criterion(outputs, moves)
            
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            
        print(f"Epoch {epoch+1} Loss: {loss.item():.4f}")
        
    torch.save(model.state_dict(), "v7p3r_chess_ai_model.pth")

if __name__ == "__main__":
    train_model()
