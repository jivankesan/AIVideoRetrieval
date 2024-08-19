import os
import torch
import torchvision
import pandas as pd
from transformers import XCLIPProcessor, XCLIPModel, TrainingArguments, Trainer
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# Define paths
data_dir = 'mp4'
train_file = 'train.csv'
val_file = 'val.csv'
test_file = 'test.csv'

# Load the CSV files
def load_csv(file_path):
    df = pd.read_csv(file_path)
    video_paths = [os.path.join(data_dir, filename) for filename in df['filename']]
    labels = df['label'].tolist()
    return video_paths, labels

train_videos, train_labels = load_csv(train_file)
val_videos, val_labels = load_csv(val_file)
test_videos, test_labels = load_csv(test_file)

# Create a custom dataset
class VideoDataset(Dataset):
    def __init__(self, video_paths, labels, processor):
        self.video_paths = video_paths
        self.labels = labels
        self.processor = processor

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        label = self.labels[idx]
        video_tensor = self.load_video(video_path)
        inputs = self.processor(video_tensor, return_tensors='pt')
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}  # Remove the batch dimension
        inputs['labels'] = torch.tensor(label)
        return inputs

    def load_video(self, video_path):
        # Implement video loading and preprocessing here
        video_tensor, _, _ = torchvision.io.read_video(video_path)
        return video_tensor

# Initialize the processor and model
processor = XCLIPProcessor.from_pretrained('microsoft/xclip-base-patch32')
model = XCLIPModel.from_pretrained('microsoft/xclip-base-patch32').to('cuda')

# Create datasets and dataloaders
train_dataset = VideoDataset(train_videos, train_labels, processor)
val_dataset = VideoDataset(val_videos, val_labels, processor)

train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=4)

# Define training arguments
training_args = TrainingArguments(
    output_dir='./results',
    evaluation_strategy='epoch',
    learning_rate=2e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=3,
    weight_decay=0.01,
    save_strategy='epoch'
)

# Custom training loop with progress bar
optimizer = torch.optim.AdamW(model.parameters(), lr=training_args.learning_rate)

def train_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0
    progress_bar = tqdm(dataloader, desc="Training")
    for batch in progress_bar:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        total_loss += loss.item()
        progress_bar.set_postfix(loss=total_loss / len(dataloader))
    return total_loss / len(dataloader)

def evaluate_epoch(model, dataloader, device):
    model.eval()
    total_loss = 0
    progress_bar = tqdm(dataloader, desc="Evaluating")
    with torch.no_grad():
        for batch in progress_bar:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            total_loss += loss.item()
            progress_bar.set_postfix(loss=total_loss / len(dataloader))
    return total_loss / len(dataloader)

# Training loop
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

for epoch in range(training_args.num_train_epochs):
    print(f"Epoch {epoch + 1}/{training_args.num_train_epochs}")
    train_loss = train_epoch(model, train_dataloader, optimizer, device)
    val_loss = evaluate_epoch(model, val_dataloader, device)
    print(f"Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f}")

# Save the model
model.save_pretrained('./results')
processor.save_pretrained('./results')