import os
import time
import wave
import numpy as np
import torch
from torch import nn
import torchaudio
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

from torch.utils.data import Dataset, DataLoader
from torchaudio.datasets import SPEECHCOMMANDS


### SETUP


data_dir = "./data"
output_dir = "./outputs"

os.makedirs(output_dir, exist_ok=True)

batch_size = 128
num_workers = 8
epochs = 20
learning_rate = 1e-3
sample_rate = 16000
num_samples = 16000

commands = [
    "yes", "no", "up", "down", "left",
    "right", "on", "off", "stop", "go"
]

num_classes = len(commands)
label_to_index = {label: i for i, label in enumerate(commands)}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Device:", device)


### Audio transforms

mel_transform = torchaudio.transforms.MelSpectrogram(
    sample_rate=sample_rate,
    n_fft=400,
    hop_length=160,
    n_mels=64
)

to_db = torchaudio.transforms.AmplitudeToDB()

# One resampler per original sample rate, created on first use
resamplers = {}


def preprocess_audio(waveform, original_sample_rate):

    # Converts stereo to mono
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # Resamples to 16 kHz if necessary
    if original_sample_rate != sample_rate:
        if original_sample_rate not in resamplers:
            resamplers[original_sample_rate] = torchaudio.transforms.Resample(
                original_sample_rate,
                sample_rate
            )
        waveform = resamplers[original_sample_rate](waveform)

    # Pad or truncate to one second
    if waveform.shape[1] < num_samples:
        waveform = torch.nn.functional.pad(
            waveform,
            (0, num_samples - waveform.shape[1])
        )
    else:
        waveform = waveform[:, :num_samples]

    # Mel spectrogram and decibel conversion
    spectrogram = mel_transform(waveform)
    spectrogram = to_db(spectrogram)

    # Standardize so that masking with 0 means "average level"
    spectrogram = (spectrogram - spectrogram.mean()) / (spectrogram.std() + 1e-5)

    return spectrogram


### DATA

class SpeechCommandsDataset(Dataset):

    def __init__(self, subset):

        self.dataset = SPEECHCOMMANDS(
            root=data_dir,
            download=True,
            subset=subset 
        )  

        # Select only the ten target commands, beyond is out of scope for now.
        self.indices = [
            i for i, path in enumerate(self.dataset._walker)
            if os.path.basename(os.path.dirname(path)) in commands
        ]

        print(f"{subset}: {len(self.indices)} samples")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):

        path = self.dataset._walker[self.indices[index]]

        # I had issues with torchcodec, so im reading wav directly
        with wave.open(path, "rb") as wav:
            sr = wav.getframerate()
            channels = wav.getnchannels()
            sample_width = wav.getsampwidth()
            num_frames = wav.getnframes()

            if sample_width != 2:
                raise ValueError(
                    f"Expected 16-bit PCM, got {sample_width * 8}-bit"
                )

            audio_bytes = wav.readframes(num_frames)

        # Convert 16-bit PCM to float32.
        audio = np.frombuffer(audio_bytes, dtype="<i2").copy()
        audio = audio.astype(np.float32) / 32768.0

        # Shape: [channels, samples]
        audio = audio.reshape(-1, channels).T
        waveform = torch.from_numpy(audio)

        # Preprocess audio
        spectrogram = preprocess_audio(waveform, sr)

        # Convert label to integer
        label = os.path.basename(os.path.dirname(path))
        label = label_to_index[label]

        return spectrogram, label


### Load datasets

training_data = SpeechCommandsDataset("training")
validation_data = SpeechCommandsDataset("validation")
test_data = SpeechCommandsDataset("testing")


def make_loader(dataset, shuffle):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=True
    )


train_dataloader = make_loader(training_data, shuffle=True)
validation_dataloader = make_loader(validation_data, shuffle=False)
test_dataloader = make_loader(test_data, shuffle=False)


## Plotting Mel spectogram for fun and sanity check

spectrogram, label = training_data[0]

plt.figure(figsize=(10, 4))

plt.imshow(
    spectrogram[0].numpy(),
    origin="lower",
    aspect="auto"
)

plt.title(f"Mel spectrogram: {commands[label]}")
plt.xlabel("Time frame")
plt.ylabel("Mel frequency bin")
plt.colorbar(label="dB")

plt.tight_layout()
plt.savefig(f"{output_dir}/spectrogram.png", dpi=150)
plt.close()


### ACTUAL MODEL STARTS HERE ###

class SpeechCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()

        self.features = nn.Sequential(
        nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
        )
        

        self.classifier = nn.Sequential(
        nn.AdaptiveAvgPool2d((2, 4)),
        nn.Flatten(),
        nn.Dropout(0.1),
        nn.Linear(128 * 2 * 4, 128), nn.ReLU(),
        nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

model = SpeechCNN(num_classes=num_classes).to(device)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
# T_max=epochs: lr decays smoothly from learning_rate to ~0 over the whole run
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)


### TRAINING AND EVALUATION ###

def train_one_epoch():
    model.train()

    total_loss = 0
    total_correct = 0
    total_samples = 0
    ### Augmentation (training only)
    freq_mask = torchaudio.transforms.FrequencyMasking(freq_mask_param=8, iid_masks=True)
    time_mask = torchaudio.transforms.TimeMasking(time_mask_param=12, iid_masks=True)

    for spectrograms, labels in train_dataloader:

        spectrograms = spectrograms.to(device)
        #spectrograms = time_mask(freq_mask(spectrograms))
        labels = labels.to(device)

        # Forward pass
        predictions = model(spectrograms)
        loss = loss_fn(predictions, labels)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # Epoch statistics
        total_loss += loss.item() * labels.size(0)
        total_correct += (predictions.argmax(dim=1) == labels).sum().item()
        total_samples += labels.size(0)

    return total_loss / total_samples, total_correct / total_samples


def evaluate(dataloader, return_predictions=False):
    model.eval()

    total_loss = 0
    total_correct = 0
    total_samples = 0

    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for spectrograms, labels in dataloader:
            spectrograms = spectrograms.to(device)
            labels = labels.to(device)

            predictions = model(spectrograms)
            predicted_labels = predictions.argmax(dim=1)

            total_loss += loss_fn(predictions, labels).item() * labels.size(0)
            total_correct += (predicted_labels == labels).sum().item()
            total_samples += labels.size(0)

            if return_predictions:
                all_predictions.extend(predicted_labels.cpu().tolist())
                all_labels.extend(labels.cpu().tolist())

    loss = total_loss / total_samples
    accuracy = total_correct / total_samples

    if return_predictions:
        return loss, accuracy, all_labels, all_predictions
    return loss, accuracy


history = {
    "tr_loss": [], "tr_acc": [],
    "va_loss": [], "va_acc": [],
    "te_loss": [], "te_acc": [],
}

#Due to the masking above, i am getting a lot of jitter, so i will use a checkpoint system.
best_val_acc = 0
best_epoch = 0
best_state = None

for epoch in range(epochs):
    t0 = time.time()

    tr = train_one_epoch()
    va = evaluate(validation_dataloader)
    te = evaluate(test_dataloader)
    scheduler.step()

    for key, value in zip(history, (*tr, *va, *te)):
        history[key].append(value)

    #the aforementioned checkpoint system
    if va[1] > best_val_acc:
        best_val_acc = va[1]
        best_epoch = epoch + 1
        best_state = {k: v.clone() for k, v in model.state_dict().items()}
        torch.save(best_state, f"{output_dir}/best_model.pt")

    print(
        f"epoch {epoch + 1:2d}/{epochs}: "
        f"train {tr[0]:.3f}/{tr[1]:.3f}  "
        f"val {va[0]:.3f}/{va[1]:.3f}  "
        f"test {te[0]:.3f}/{te[1]:.3f}  "
        f"({time.time() - t0:.1f}s)"
    )

###Loads the best checkpoint
model.load_state_dict(best_state)
print(f"Loaded best model from epoch {best_epoch} (val acc {100 * best_val_acc:.2f}%)")

### Curves

fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4))

for name, key in [("train", "tr"), ("validation", "va"), ("test", "te")]:
    a1.plot(history[f"{key}_loss"], label=name)
    a2.plot(history[f"{key}_acc"], label=name)

a1.set_xlabel("epoch")
a1.set_ylabel("loss")
a1.legend()
a2.set_xlabel("epoch")
a2.set_ylabel("accuracy")
a2.legend()

plt.tight_layout()
plt.savefig(f"{output_dir}/curves.png", dpi=150)
plt.close()


### Confusion matrix

test_loss, test_acc, all_labels, all_predictions = evaluate(
    test_dataloader, return_predictions=True
)

print(f"Final test accuracy: {100 * test_acc:.2f}%")

cm = confusion_matrix(
    all_labels,
    all_predictions,
    labels=range(num_classes)
)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=commands
)

disp.plot(cmap="Blues", xticks_rotation=45)

plt.title("Test Set Confusion Matrix")
plt.tight_layout()
plt.savefig(f"{output_dir}/confusion_matrix.png", dpi=300)
plt.close()