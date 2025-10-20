import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
from sklearn.model_selection import train_test_split

# ====================== CONFIG ======================
DATA_DIR = "./data_train"   # parent folder containing images
BATCH_SIZE = 32
LR = 1e-4
EPOCHS = 10
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_SAVE = "puzzle_evaluator.pth"
IMG_SIZE = 224
# ====================================================


# ======== Dataset definition =========
class PuzzleEvalDataset(Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform or transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        img = Image.open(path).convert("RGB")

        # Label by filename pattern
        label = 1.0 if "original" in os.path.basename(path).lower() else 0.0

        img = self.transform(img)
        return img, torch.tensor(label, dtype=torch.float32)


# ======== Model definition (ResNet18-based) =========
class PuzzleEvaluator(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        self.backbone = models.resnet18(pretrained=pretrained)
        in_feats = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Linear(in_feats, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.backbone(x).squeeze(1)


# ======== Data loading =========
def load_image_paths(data_dir):
    exts = (".jpg", ".png", ".jpeg")
    all_paths = [os.path.join(data_dir, f)
                 for f in os.listdir(data_dir)
                 if f.lower().endswith(exts)]
    # Shuffle and split
    train_paths, val_paths = train_test_split(all_paths, test_size=0.1, random_state=42)
    return train_paths, val_paths


# ======== Training loop =========
def train():
    train_paths, val_paths = load_image_paths(DATA_DIR)
    print(f"Found {len(train_paths)} train images, {len(val_paths)} val images")

    train_dataset = PuzzleEvalDataset(train_paths)
    val_dataset = PuzzleEvalDataset(val_paths)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    model = PuzzleEvaluator(pretrained=True).to(DEVICE)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    best_val_acc = 0.0
    for epoch in range(EPOCHS):
        model.train()
        total_loss, total_correct, total_samples = 0, 0, 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            preds = model(imgs)
            loss = criterion(preds, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * imgs.size(0)
            total_correct += ((preds > 0.5) == (labels > 0.5)).sum().item()
            total_samples += imgs.size(0)

        train_acc = total_correct / total_samples
        val_acc, val_loss = evaluate(model, val_loader, criterion)

        print(f"Epoch {epoch+1}/{EPOCHS} | "
              f"Train Acc: {train_acc:.3f} | Val Acc: {val_acc:.3f} | Val Loss: {val_loss:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), MODEL_SAVE)
            print(f"✅ Saved best model with val_acc={best_val_acc:.3f}")

    print(f"Training complete. Best validation accuracy: {best_val_acc:.3f}")


# ======== Evaluation helper =========
def evaluate(model, loader, criterion):
    model.eval()
    total_loss, total_correct, total_samples = 0, 0, 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            preds = model(imgs)
            loss = criterion(preds, labels)
            total_loss += loss.item() * imgs.size(0)
            total_correct += ((preds > 0.5) == (labels > 0.5)).sum().item()
            total_samples += imgs.size(0)
    return total_correct / total_samples, total_loss / total_samples


# ======== Main =========
if __name__ == "__main__":
    train()
