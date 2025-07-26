from chess_core import *
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset
import yaml
import numpy as np
import os
from pathlib import Path

torch.backends.cudnn.benchmark = True  # Enable CuDNN auto-tuner

# Load configuration
with open("config.yaml", encoding='utf-8-sig') as f:  # utf-8-sig handles BOM
    config = yaml.safe_load(f)

# Debug: Print config structure
print(f"🔧 Config loaded: {list(config.keys()) if config else 'None'}")
if config and 'training' in config:
    print(f"🔧 Training config: {config['training']}")
else:
    print("❌ No 'training' key found in config!")

# Calculate move weights based on frequency
def calculate_move_weights(moves_list, num_classes):
    counts = np.bincount(moves_list, minlength=num_classes)
    weights = 1.0 / (np.sqrt(counts) + 1e-6)  # Inverse sqrt frequency
    return torch.tensor(weights / weights.max(), dtype=torch.float32)

# Initialize move-to-index mapping
def create_move_vocab(datasets):
    """Create vocabulary from multiple datasets"""
    all_moves = []
    for dataset in datasets:
        all_moves.extend(dataset.moves)
    return {move: idx for idx, move in enumerate(np.unique(all_moves))}

def load_enhanced_datasets():
    """Load the user's games plus all training position files with smart filtering"""
    datasets = []
    
    # Load user's personal games (main training data) - no limit for user's own games
    print("📚 Loading user's personal games...")
    user_dataset = ChessDataset("games.pgn")  # Will auto-detect v7p3r games and filter
    datasets.append(user_dataset)
    print(f"   ✅ Loaded {len(user_dataset)} user positions (v7p3r moves only)")
    
    # Load all additional training PGN files with 100 game limit per file
    training_dir = Path("training_positions")
    if training_dir.exists():
        print("\n📖 Loading additional chess knowledge (max 100 games per file)...")
        pgn_files = list(training_dir.glob("*.pgn"))
        
        for pgn_file in pgn_files:
            try:
                print(f"   📄 Loading {pgn_file.name} (max 100 games)...")
                dataset = ChessDataset(str(pgn_file), max_games=100)  # Limit to 100 games per file
                if len(dataset) > 0:
                    datasets.append(dataset)
                    print(f"      ✅ Added {len(dataset)} positions")
                else:
                    print(f"      ⚠️  No positions found")
            except Exception as e:
                print(f"      ❌ Error loading {pgn_file.name}: {str(e)}")
    
    print(f"\n🎯 Total datasets loaded: {len(datasets)}")
    total_positions = sum(len(d) for d in datasets)
    print(f"🎯 Total training positions: {total_positions:,}")
    print("🧠 Training strategy:")
    print("   • Games with v7p3r: Only v7p3r moves trained (preserves personality)")
    print("   • Games without v7p3r: All moves trained (gains chess knowledge)")
    
    return datasets

def train_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔥 Training on device: {device}")
    
    # Toggle to skip vocab generation if vocab already exists
    REGENERATE_VOCAB = False  # Set to True to regenerate vocab from scratch
    
    # Load all datasets (user games + additional training data)
    datasets = load_enhanced_datasets()
    
    # Create or load move vocabulary
    if REGENERATE_VOCAB or not os.path.exists("move_vocab.pkl"):
        print("\n🔤 Creating enhanced move vocabulary...")
        move_to_index = create_move_vocab(datasets)
        num_classes = len(move_to_index)
        print(f"   📊 Enhanced vocabulary size: {num_classes:,} unique moves")
        
        # Save the enhanced move vocabulary
        import pickle
        with open("move_vocab.pkl", "wb") as f:
            pickle.dump(move_to_index, f)
        print("   💾 Enhanced vocabulary saved to move_vocab.pkl")
    else:
        print("\n🔤 Loading existing move vocabulary...")
        import pickle
        with open("move_vocab.pkl", "rb") as f:
            move_to_index = pickle.load(f)
        num_classes = len(move_to_index)
        print(f"   📊 Loaded vocabulary size: {num_classes:,} unique moves")
    
    # Convert string moves to indices for all datasets
    print("\n🔄 Converting moves to indices...")
    processed_datasets = []
    for i, dataset in enumerate(datasets):
        try:
            move_indices = [move_to_index[move] for move in dataset.moves if move in move_to_index]
            dataset.moves = move_indices
            if len(dataset.moves) > 0:
                processed_datasets.append(dataset)
                print(f"   ✅ Dataset {i+1}: {len(dataset.moves)} moves converted")
            else:
                print(f"   ⚠️  Dataset {i+1}: No valid moves found")
        except Exception as e:
            print(f"   ❌ Dataset {i+1} conversion error: {str(e)}")
    
    # Combine all datasets
    if processed_datasets:
        combined_dataset = ConcatDataset(processed_datasets)
        print(f"🎯 Combined dataset size: {len(combined_dataset):,} positions")
        
        # Collect all moves for weight calculation
        all_moves_for_weights = []
        for dataset in processed_datasets:
            all_moves_for_weights.extend(dataset.moves)
    else:
        raise ValueError("No valid datasets processed!")
    
    dataloader = DataLoader(
        combined_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    model = ChessAI(num_classes).to(device)
    
    # Calculate move weights from all collected moves
    all_moves_for_weights = []
    for dataset in processed_datasets:
        all_moves_for_weights.extend(dataset.moves)
    
    criterion = nn.CrossEntropyLoss(
        weight=calculate_move_weights(np.array(all_moves_for_weights), num_classes).to(device)
    )
    
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )
    
    print("\n🚀 Starting enhanced training with knowledge injection...")
    print(f"   📊 Model vocabulary: {num_classes:,} unique moves")
    print(f"   📦 Batch size: {config['training']['batch_size']}")
    print(f"   🎯 Epochs: {config['training']['epochs']}")
    print(f"   🧠 Learning rate: {config['training']['learning_rate']}")
    
    for epoch in range(config['training']['epochs']):
        model.train()
        total_loss = 0
        batch_count = 0
        print(f"\n   🚀 Starting Epoch {epoch+1}/{config['training']['epochs']}...")
        
        for batch_idx, (positions, moves) in enumerate(dataloader):
            positions = positions.to(device, non_blocking=True)
            moves = torch.tensor(moves, dtype=torch.long).to(device, non_blocking=True)
            
            outputs = model(positions)
            loss = criterion(outputs, moves)
            
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            total_loss += loss.item()
            batch_count += 1
            
            # Progress update every 100 batches
            if batch_idx % 100 == 0:
                current_loss = loss.item()
                print(f"      Batch {batch_idx}: Loss = {current_loss:.4f}")
            
        avg_loss = total_loss / batch_count if batch_count > 0 else 0
        print(f"   🔥 Epoch {epoch+1}/{config['training']['epochs']} - Average Loss: {avg_loss:.4f} ({batch_count} batches)")
        
        # Early stopping if loss becomes very low
        if avg_loss < 0.01:
            print("   🎯 Early stopping: Loss sufficiently low")
            break
        
    print("\n💾 Saving enhanced model...")
    torch.save(model.state_dict(), "v7p3r_chess_ai_model.pth")
    print("✅ Enhanced training complete!")
    print("🎉 V0.5.31 AI now has your personality + enhanced chess knowledge!")

if __name__ == "__main__":
    train_model()
