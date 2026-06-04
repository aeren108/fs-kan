import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from modelnet_dataloader import get_modelnet_loaders
from fskan import FSKANClassifier, StandardKANClassifier

def parse_args():
    parser = argparse.ArgumentParser(description="Train KAN/FS-KAN models on ModelNet40")
    parser.add_argument("--model", type=str, default="fskan_std", 
                        choices=["fskan_std", "fskan_eff", "kan_std"],
                        help="Model architecture to train")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--num_points", type=int, default=1024, help="Number of points per object")
    parser.add_argument("--hidden_dim", type=int, default=36, help="Hidden dimension size")
    parser.add_argument("--num_equiv_layers", type=int, default=2, help="Number of equivariant layers (for FS-KAN)")
    parser.add_argument("--train-set-size", dest="train_set_size", type=int, default=None, help="Number of samples to use from the train set")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--save_dir", type=str, default="checkpoints", help="Directory to save model checkpoints")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Set seed for reproducibility
    torch.manual_seed(args.seed)
    
    # Device setup
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")
    
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Load Data
    print("Initializing DataLoaders...")
    train_loader, test_loader = get_modelnet_loaders(
        batch_size=args.batch_size,
        num_points=args.num_points,
        use_augmentation=False,
        train_set_size=args.train_set_size
    )
    
    # Initialize Model
    print(f"Initializing model: {args.model}")
    if args.model == "fskan_std":
        model = FSKANClassifier(
            in_features=3, 
            hidden_dim=args.hidden_dim, 
            num_classes=40,
            num_equiv_layers=args.num_equiv_layers, 
            use_efficient=False, 
            pool="sum"
        )
    elif args.model == "fskan_eff":
        model = FSKANClassifier(
            in_features=3, 
            hidden_dim=args.hidden_dim, 
            num_classes=40,
            num_equiv_layers=args.num_equiv_layers, 
            use_efficient=True, 
            pool="sum"
        )
    elif args.model == "kan_std":
        model = StandardKANClassifier(
            num_points=args.num_points, 
            in_features=3,
            hidden_dims=[args.hidden_dim, args.hidden_dim], 
            num_classes=40
        )
    else:
        raise ValueError(f"Unknown model type: {args.model}")
        
    model = model.to(device)
    
    # Optimizer & Loss
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss()
    
    best_acc = 0.0
    
    # Training Loop
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs} [Train]")
        for batch in train_pbar:
            # batch.pos is (B * num_points, 3). Reshape to (B, num_points, 3)
            # PyG batches node features into a flattened tensor
            B = batch.batch.max().item() + 1
            N = args.num_points
            
            x = batch.pos.view(B, N, 3).to(device)
            y = batch.y.to(device)
            
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * B
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += B
            
            train_pbar.set_postfix({"loss": f"{loss.item():.4f}", "acc": f"{100.*correct/total:.2f}%"})
            
        scheduler.step()
        train_acc = 100. * correct / total
        train_loss = total_loss / total
        
        # Evaluation Loop
        model.eval()
        test_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            test_pbar = tqdm(test_loader, desc=f"Epoch {epoch}/{args.epochs} [Test ]")
            for batch in test_pbar:
                B = batch.batch.max().item() + 1
                N = args.num_points
                
                x = batch.pos.view(B, N, 3).to(device)
                y = batch.y.to(device)
                
                logits = model(x)
                loss = criterion(logits, y)
                
                test_loss += loss.item() * B
                preds = logits.argmax(dim=1)
                correct += (preds == y).sum().item()
                total += B
                
                test_pbar.set_postfix({"loss": f"{loss.item():.4f}", "acc": f"{100.*correct/total:.2f}%"})
                
        test_acc = 100. * correct / total
        test_loss = test_loss / total
        
        print(f"--> Epoch {epoch} Summary | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}%")
        
        if test_acc > best_acc:
            best_acc = test_acc
            save_path = os.path.join(args.save_dir, f"best_{args.model}.pth")
            print(f"*** New best accuracy: {best_acc:.2f}%. Saving model to {save_path} ***")
            torch.save(model.state_dict(), save_path)
            
    print(f"\nTraining Complete. Best Test Accuracy: {best_acc:.2f}%")

if __name__ == "__main__":
    main()
