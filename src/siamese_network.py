import argparse
import glob
import json
import os
import random
import time

import matplotlib.pyplot as plt
import numpy as np
import timm
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms

# Configuration Defaults
IMAGE_EXTS = ["*.jpg", "*.jpeg", "*.png", "*.webp"]
DEFAULT_IMAGE_DIR = "./images"
DEFAULT_RESULTS_DIR = "./results"
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
LEARNING_RATE = 0.001
EPOCHS = 50
DEV_DATASET_SIZE = 10000


class AdjacencyDataset(Dataset):
    """
    Dataset that generates pairs of image patches on the fly.
    Label 1: Left and Right halves of the same image (Adjacent).
    Label 0: Left half of Image A, Left half of Image B (Non-adjacent/Random).
    """

    def __init__(self, image_dir, transform=None, dev_mode=False):
        self.image_files = []
        for ext in IMAGE_EXTS:
            self.image_files.extend(glob.glob(os.path.join(image_dir, ext)))

        # In DEV_MODE, limit to 1000 images
        if dev_mode:
            print(
                f"[DEV_MODE] Limiting dataset to {DEV_DATASET_SIZE} images (found {len(self.image_files)})"
            )
            self.image_files = self.image_files[:DEV_DATASET_SIZE]
        else:
            print(f"Found {len(self.image_files)} images.")

        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # 1. Load the primary image (Image A)
        img_path_a = self.image_files[idx]
        image_a = Image.open(img_path_a).convert("RGB")
        w, h = image_a.size

        # 2. Extract Left and Right patches
        # We define "adjacent" as the vertical midline split
        # Left Patch: 0 to w/2
        # Right Patch: w/2 to w
        mid_x = w // 2

        patch_left_a = image_a.crop((0, 0, mid_x, h))
        patch_right_a = image_a.crop((mid_x, 0, w, h))

        # 3. Determine if Positive or Negative Example
        # 50% chance of being a positive pair (Adjacent)
        # 50% chance of being a negative pair (Random)
        is_positive = random.random() > 0.5

        label = 1.0 if is_positive else 0.0

        if is_positive:
            patch_1 = patch_left_a
            patch_2 = patch_right_a
        else:
            # Negative: Pair Left patch of A with Left patch of random Image B
            rand_idx = random.randint(0, len(self.image_files) - 1)
            while rand_idx == idx and len(self.image_files) > 1:
                rand_idx = random.randint(0, len(self.image_files) - 1)

            img_path_b = self.image_files[rand_idx]
            image_b = Image.open(img_path_b).convert("RGB")
            w_b, h_b = image_b.size
            mid_x_b = w_b // 2

            # Using left patch of B to ensure it doesn't accidentally march right patch of A if they were same image
            patch_left_b = image_b.crop((0, 0, mid_x_b, h_b))

            patch_1 = patch_left_a
            patch_2 = patch_left_b

        # 4. Apply Transforms (Resize -> Tensor -> Normalize)
        target_w, target_h = IMG_SIZE

        # patch_1 is the Left patch. Adjacency boundary is on its RIGHT.
        if patch_1.width > target_w:
            patch_1 = patch_1.crop(
                (patch_1.width - target_w, 0, patch_1.width, patch_1.height)
            )

        # patch_2 is the Right patch (or serves that role). Adjacency boundary is on its LEFT.
        if patch_2.width > target_w:
            patch_2 = patch_2.crop((0, 0, target_w, patch_2.height))

        if self.transform:
            patch_1 = self.transform(patch_1)
            patch_2 = self.transform(patch_2)

        return patch_1, patch_2, torch.tensor(label, dtype=torch.float32)


class SiameseNetwork(nn.Module):
    """
    Siamese Network with configurable backbone (ResNet18 or timm models).
    """

    def __init__(self, model_name="resnet18"):
        super(SiameseNetwork, self).__init__()

        self.model_name = model_name

        # Store input dimensions
        self.input_size = IMG_SIZE  # (224, 224)
        self.input_channels = 3

        print(f"Loading {model_name} from timm...")
        # create_model with num_classes=0 gives the features directly
        self.backbone = timm.create_model(model_name, pretrained=True, num_classes=0)

        # Determine feature dimension dynamically
        with torch.no_grad():
            dummy_input = torch.zeros(
                1, self.input_channels, self.input_size[0], self.input_size[1]
            )
            dummy_output = self.backbone(dummy_input)
            # Handle diff output types (some backbones return (B, C, 1, 1), some (B, C))
            if len(dummy_output.shape) == 4:
                dummy_output = dummy_output.view(dummy_output.size(0), -1)
            feature_dim = dummy_output.shape[1]

        print(f"Detected feature dimension: {feature_dim}")
        print(
            f"Input size: {self.input_channels} x {self.input_size[0]} x {self.input_size[1]}"
        )

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim * 2, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward_one(self, x):
        # x shape: (Batch, 3, H, W)
        x = self.backbone(x)

        # ResNet backbone from torchvision returns (Batch, 512, 1, 1)
        # timm models with num_classes=0 usually return (Batch, Features)
        if len(x.shape) == 4:
            x = x.view(x.size(0), -1)

        return x

    def forward(self, x1, x2):
        feat1 = self.forward_one(x1)
        feat2 = self.forward_one(x2)

        # Concatenate features
        combined = torch.cat((feat1, feat2), dim=1)

        output = self.classifier(combined)
        return output


def save_validation_plot(results_dir, model, dataloader, epoch, device):
    """
    Saves a plot of a few validation examples with predictions.
    """
    model.eval()
    images1, images2, labels = next(iter(dataloader))

    # Move to device
    images1 = images1.to(device)
    images2 = images2.to(device)
    labels = labels.to(device)

    with torch.no_grad():
        outputs = model(images1, images2)

    # Plot first 4 samples
    fig, axes = plt.subplots(4, 1, figsize=(6, 12))
    for i in range(min(4, len(labels))):
        img1 = images1[i].cpu().numpy().transpose((1, 2, 0))
        img2 = images2[i].cpu().numpy().transpose((1, 2, 0))

        # Un-normalize for display (approximate)
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img1 = std * img1 + mean
        img2 = std * img2 + mean
        img1 = np.clip(img1, 0, 1)
        img2 = np.clip(img2, 0, 1)

        # Concatenate for display with a small black gap
        h, w, c = img1.shape
        gap = np.zeros((h, 10, c))
        combined_img = np.hstack((img1, gap, img2))

        lbl = labels[i].item()
        pred = outputs[i].item()

        ax = axes[i] if isinstance(axes, np.ndarray) else axes
        ax.imshow(combined_img)
        ax.set_title(f"Label: {int(lbl)} (Adjacent=1), Pred: {pred:.3f}")
        ax.axis("off")

    plt.tight_layout()
    out_file = f"epoch_{epoch}_results.png"
    plt.savefig(os.path.join(results_dir, out_file))
    print(f"Saved visualization to {out_file}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Train Adjacency Detection Model")
    parser.add_argument(
        "--images", type=str, default=DEFAULT_IMAGE_DIR, help="Path to image directory"
    )
    parser.add_argument(
        "--results",
        type=str,
        default=DEFAULT_RESULTS_DIR,
        help="Path to results directory",
    )
    parser.add_argument(
        "--dev",
        action="store_true",
        help=f"Run in DEV_MODE ({DEV_DATASET_SIZE} images)",
    )
    parser.add_argument("--epochs", type=int, default=EPOCHS, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE, help="Batch size")
    parser.add_argument(
        "--learning_rate", type=float, default=LEARNING_RATE, help="Learning rate"
    )
    parser.add_argument(
        "--patience", type=int, default=10, help="Patience for early stopping"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="resnet18",
        help="Model architecture (resnet18 or timm model name)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=os.cpu_count() or 4,
        help="Number of data loading workers (default: all CPU cores)",
    )
    args = parser.parse_args()

    # Check device
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using device: {device}")

    # Create results directory if it doesn't exist
    os.makedirs(args.results, exist_ok=True)

    # Transforms
    transform = transforms.Compose(
        [
            transforms.Resize(IMG_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # Dataset
    full_dataset = AdjacencyDataset(args.images, transform=transform, dev_mode=args.dev)

    if len(full_dataset) == 0:
        print(f"No images found in {args.images}. Please check path.")
        return

    # Split
    total_size = len(full_dataset)
    val_size = min(int(0.1 * total_size), 10000)
    test_size = min(int(0.1 * total_size), 10000)
    train_size = total_size - val_size - test_size

    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset, [train_size, val_size, test_size]
    )

    print(
        f"Dataset split: Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_dataset)}"
    )

    use_pin_memory = device.type in ["cuda", "mps"]
    use_persistent_workers = args.workers > 0

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=use_pin_memory,
        persistent_workers=use_persistent_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=use_pin_memory,
        persistent_workers=use_persistent_workers,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=use_pin_memory,
        persistent_workers=use_persistent_workers,
    )

    # Model
    model = SiameseNetwork(model_name=args.model).to(device)

    # Criterion & Optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    # Training Loop
    best_val_loss = float("inf")
    patience_counter = 0
    start_time = time.time()

    epochs_run = 0
    for epoch in range(args.epochs):
        epochs_run = epoch + 1
        model.train()
        running_loss = 0.0

        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        for i, (img1, img2, labels) in enumerate(train_loader):
            img1, img2, labels = img1.to(device), img2.to(device), labels.to(device)
            labels = labels.unsqueeze(1)  # Match output shape (Batch, 1)

            optimizer.zero_grad()
            outputs = model(img1, img2)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if i % 10 == 0:
                print(f"  Step {i}/{len(train_loader)}, Loss: {loss.item():.4f}")

        avg_train_loss = running_loss / len(train_loader)

        # Validation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for img1, img2, labels in val_loader:
                img1, img2, labels = img1.to(device), img2.to(device), labels.to(device)
                labels = labels.unsqueeze(1)

                outputs = model(img1, img2)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                predicted = (outputs > 0.5).float()
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        avg_val_loss = val_loss / len(val_loader)
        acc = 100 * correct / total

        print(f"Epoch {epoch + 1} Summary:")
        print(f"  Train Loss: {avg_train_loss:.4f}")
        print(f"  Val Loss: {avg_val_loss:.4f}")
        print(f"  Val Acc: {acc:.2f}%")

        # Visualization
        save_validation_plot(args.results, model, val_loader, epoch + 1, device)

        # Save Best Model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(
                model.state_dict(),
                os.path.join(args.results, "model.pth"),
            )
            print("  Saved best model.")
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"  No improvement. Patience: {patience_counter}/{args.patience}")

        if patience_counter >= args.patience:
            print(f"\nEarly stopping triggered after {epoch + 1} epochs.")
            break

    end_time = time.time()
    training_time = end_time - start_time

    # Final Eval on Test Set
    print("\nRunning Final Evaluation on Test Set...")
    model.load_state_dict(torch.load(os.path.join(args.results, "model.pth")))
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    tp = 0
    tn = 0
    fp = 0
    fn = 0

    with torch.no_grad():
        for img1, img2, labels in test_loader:
            img1, img2, labels = img1.to(device), img2.to(device), labels.to(device)
            labels = labels.unsqueeze(1)

            outputs = model(img1, img2)
            loss = criterion(outputs, labels)
            test_loss += loss.item()

            predicted = (outputs > 0.5).float()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            tp += ((predicted == 1) & (labels == 1)).sum().item()
            tn += ((predicted == 0) & (labels == 0)).sum().item()
            fp += ((predicted == 1) & (labels == 0)).sum().item()
            fn += ((predicted == 0) & (labels == 1)).sum().item()

    avg_test_loss = test_loss / len(test_loader)
    test_acc = 100 * correct / total

    # Calculate additional metrics
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * (precision * recall) / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    print("Test Set Results:")
    print(f"  Test Loss: {avg_test_loss:.4f}")
    print(f"  Test Acc: {test_acc:.2f}%")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1 Score:  {f1:.4f}")

    # Save metrics
    metrics = {
        "test_loss": avg_test_loss,
        "test_acc": test_acc,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "training_time": training_time,
        "epochs_run": epochs_run,
        "parameters": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "model": args.model,
            "patience": args.patience,
        },
    }
    with open(os.path.join(args.results, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)
    print(f"Saved metrics to {os.path.join(args.results, 'metrics.json')}")


if __name__ == "__main__":
    main()
