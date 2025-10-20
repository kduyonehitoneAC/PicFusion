# train_edge_cnn.py
import os, glob, random
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split

# ===== CONFIG =====
ROWS, COLS = 3, 5
IMAGE_HEIGHT, IMAGE_WIDTH = 360, 600
PATCH_H, PATCH_W = IMAGE_HEIGHT // ROWS, IMAGE_WIDTH // COLS
EDGE_STRIP = 8
EMBED_DIM = 128
BATCH_SIZE = 64
EPOCHS = 50
LR = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGE_DIR = "./train_data"

# ===== DATASET =====
def extract_edges_from_image(img_path):
    img = Image.open(img_path).convert('RGB').resize((IMAGE_WIDTH, IMAGE_HEIGHT))
    arr = np.array(img) / 255.0
    patches = [
        arr[r*PATCH_H:(r+1)*PATCH_H, c*PATCH_W:(c+1)*PATCH_W, :]
        for r in range(ROWS) for c in range(COLS)
    ]
    edges = []
    for r in range(ROWS):
        for c in range(COLS):
            idx = r * COLS + c
            # Right–Left pair (vertical strips)
            if c < COLS - 1:
                left_edge = patches[idx][:, -EDGE_STRIP:, :]
                right_edge = patches[idx+1][:, :EDGE_STRIP, :]
                # cả hai có shape [PATCH_H, EDGE_STRIP, 3] -> [120,8,3]
                edges.append(("R-L", left_edge, right_edge, 1))
            # Bottom–Top pair (horizontal strips) → rotate để cùng hướng
            if r < ROWS - 1:
                bottom_edge = patches[idx][-EDGE_STRIP:, :, :]
                top_edge = patches[idx+COLS][:EDGE_STRIP, :, :]
                # Xoay 90 độ để thành vertical strip [PATCH_W, EDGE_STRIP, 3]
                bottom_edge_rot = np.rot90(bottom_edge, k=1, axes=(0,1))
                top_edge_rot = np.rot90(top_edge, k=1, axes=(0,1))
                edges.append(("B-T", bottom_edge_rot, top_edge_rot, 1))
    # Negative pairs (random)
    for _ in range(len(edges)):
        a, b = random.sample(patches, 2)
        left_edge = a[:, -EDGE_STRIP:, :]
        right_edge = b[:, :EDGE_STRIP, :]
        edges.append(("NEG", left_edge, right_edge, 0))
    return edges


class EdgePairDataset(Dataset):
    def __init__(self, image_dir):
        all_paths = glob.glob(os.path.join(image_dir, "*.jpg")) + glob.glob(os.path.join(image_dir, "*.png"))
        all_edges = []
        for p in tqdm(all_paths, desc="Building edge pairs"):
            all_edges.extend(extract_edges_from_image(p))
        self.data = all_edges

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        _, e1, e2, label = self.data[idx]
        e1 = np.ascontiguousarray(e1.copy())  # đảm bảo không còn stride âm
        e2 = np.ascontiguousarray(e2.copy())
        e1 = torch.tensor(e1.transpose(2, 0, 1), dtype=torch.float32)
        e2 = torch.tensor(e2.transpose(2, 0, 1), dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.float32)
        return e1, e2, label


# ===== MODEL =====
class EdgeCNN(nn.Module):
    def __init__(self, embedding_dim=EMBED_DIM):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )
        self.fc = nn.Sequential(
            nn.Linear(128, embedding_dim),
            nn.LayerNorm(embedding_dim)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

def contrastive_loss(e1, e2, label, margin=1.0):
    dist = F.pairwise_distance(e1, e2)
    loss = (label * dist**2 + (1 - label) * F.relu(margin - dist)**2).mean()
    return loss

# ===== TRAIN =====
def train_model():
    ds = EdgePairDataset(IMAGE_DIR)
    val_size = int(0.1 * len(ds))
    train_ds, val_ds = random_split(ds, [len(ds)-val_size, val_size])
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

    model = EdgeCNN().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)

    best_val = 9999
    for epoch in range(1, EPOCHS+1):
        model.train()
        tloss = 0
        for e1, e2, label in tqdm(train_loader, desc=f"Epoch {epoch}"):
            e1, e2, label = e1.to(DEVICE), e2.to(DEVICE), label.to(DEVICE)
            opt.zero_grad()
            z1, z2 = model(e1), model(e2)
            loss = contrastive_loss(z1, z2, label)
            loss.backward()
            opt.step()
            tloss += loss.item()
        tloss /= len(train_loader)

        # validation
        model.eval()
        with torch.no_grad():
            vloss = 0
            for e1, e2, label in val_loader:
                e1, e2, label = e1.to(DEVICE), e2.to(DEVICE), label.to(DEVICE)
                z1, z2 = model(e1), model(e2)
                vloss += contrastive_loss(z1, z2, label).item()
            vloss /= len(val_loader)
        print(f"Epoch {epoch}: TrainLoss={tloss:.4f}, ValLoss={vloss:.4f}")

        if vloss < best_val:
            best_val = vloss
            torch.save(model.state_dict(), "edge_cnn_best.pth")
            print("✅ Saved best model")

if __name__ == "__main__":
    train_model()
