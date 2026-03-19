import torch
import torch.nn as nn
from transformers import (
    Wav2Vec2FeatureExtractor,
    Wav2Vec2ForSequenceClassification,
    Wav2Vec2Processor
)
import librosa
import numpy as np


# Emotion categories for 6-class Hackathon demo
# Extrapolates standard 4 classes (Neutral, Happy, Angry, Sad) via blending
EMOTION_LABELS = ["Neutral", "Happy", "Angry", "Sad", "Fearful", "Surprised"]


class EmotionDetectionModel:
    def __init__(self, model_name="superb/wav2vec2-base-superb-er"):
        """
        Initialize pre-trained Wav2Vec2 model for emotion recognition.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.processor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
        self.model = Wav2Vec2ForSequenceClassification.from_pretrained(model_name)

        self.model.to(self.device)
        self.model.eval()

    def predict(self, audio_array, sr=16000):
        """
        Run inference on audio input and return predicted emotion.
        Maps 4 original classes to 6 classes based on probabilistic blending and exact audio heuristics.
        """
        # Ensure statelessness by copying the raw array
        audio_array = np.copy(audio_array)

        # Preprocessing: Convert stereo to mono
        if len(audio_array.shape) > 1:
            audio_array = audio_array.mean(axis=1)

        # Preprocessing: Trim dead silence
        audio_array, _ = librosa.effects.trim(audio_array, top_db=30)
        
        # Ensure minimum length for consistency
        if len(audio_array) < sr * 1.0:
            pad_length = int(sr * 1.0) - len(audio_array)
            audio_array = np.pad(audio_array, (0, pad_length), mode='constant')

        # Audio-Based Heuristics: Calculate low energy and variance BEFORE normalization
        rms_energy = np.mean(librosa.feature.rms(y=audio_array))
        variance = np.var(audio_array)

        # Preprocessing: Amplitude normalization
        audio_array = np.asarray(audio_array, dtype=np.float32)
        max_val = np.max(np.abs(audio_array))
        if max_val > 0:
            audio_array = audio_array / max_val

        # Features Extraction
        inputs = self.processor(
            audio_array,
            sampling_rate=sr,
            return_tensors="pt",
            padding=True
        )

        with torch.no_grad():
            inputs = {k: v.clone().detach().to(self.device) for k, v in inputs.items()}
            outputs = self.model(**inputs, output_attentions=False, output_hidden_states=False)
            
            logits = outputs.logits.clone().detach().cpu()

            # 3. Probability Calibration: Apply temperature scaling for soft predictions
            temperature = 1.5 
            logits = logits / temperature

            # Extract base 4-class probabilities (Neutral, Happy, Angry, Sad)
            probs_4 = torch.nn.functional.softmax(logits, dim=-1).squeeze().numpy()

            # 1. Class Mapping: Map 4 bases into 6 target classes smoothly
            probs_6 = np.zeros(6, dtype=np.float32)
            
            # Shift bases (leaving ~10-20% for synthetic blending)
            probs_6[0] = probs_4[0] * 0.90  # Neutral
            probs_6[1] = probs_4[1] * 0.80  # Happy
            probs_6[2] = probs_4[2] * 0.85  # Angry
            probs_6[3] = probs_4[3] * 0.85  # Sad
            
            # Synthetic Class: Fearful (Blending Sad + Angry characteristics)
            probs_6[4] = (probs_4[2] * 0.15) + (probs_4[3] * 0.15)
            # Synthetic Class: Surprised (Blending Happy + Neutral excitation)
            probs_6[5] = (probs_4[1] * 0.20) + (probs_4[0] * 0.10)

            # 2. & 4. Audio-Based Heuristics and Anti-Bias Correction
            # Low energy / flat audio strongly biases towards "Neutral"
            if rms_energy < 0.015 or variance < 0.0005:
                probs_6[0] += 0.35
                probs_6[2] *= 0.4   # Heavily penalize Angry
                probs_6[1] *= 0.5   # Penalize Happy
                probs_6[5] *= 0.5   # Penalize Surprised
                
            # High energy thresholding avoids false flat readings
            elif rms_energy > 0.08:
                probs_6[2] += 0.15  # Boost Angry
                probs_6[5] += 0.10  # Boost Surprised

            # 6. Smoothing: Redistribute slight probabilities to avoid sharp 0s
            probs_6 = np.maximum(probs_6, 0.02)
            
            # Normalize to valid distribution
            probs_6 = probs_6 / np.sum(probs_6)

            # 7. Confidence Logic & Fallbacks
            pred_idx = np.argmax(probs_6)
            confidence = probs_6[pred_idx]
            
            # Fallback to Neutral under low confidence (skewed or chaotic distribution)
            if confidence < 0.35:
                probs_6[0] += 0.2  
                probs_6 = probs_6 / np.sum(probs_6)
                pred_idx = 0
                confidence = probs_6[pred_idx]

            emotion = EMOTION_LABELS[pred_idx]

            # 8. Debugging Output Ensure variation across inputs
            print(f"[DEBUG] Audio RMS Energy: {rms_energy:.4f} | Audio Variance: {variance:.5f}")
            print(f"[DEBUG] Base 4-Class Softmax: {np.round(probs_4, 3)}")
            print(f"[DEBUG] Final 6-Class Blend: {np.round(probs_6, 3)}")
            
            # GC Cleanup
            del outputs, logits
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return emotion, float(confidence), probs_6.tolist()


# -------------------- DATASET --------------------
class EmotionDataset(torch.utils.data.Dataset):
    def __init__(self, file_paths, labels, processor, target_sr=16000):
        self.file_paths = file_paths
        self.labels = labels
        self.processor = processor
        self.target_sr = target_sr

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        audio, sr = librosa.load(self.file_paths[idx], sr=self.target_sr)

        inputs = self.processor(
            audio,
            sampling_rate=self.target_sr,
            return_tensors="pt",
            padding="max_length",
            max_length=self.target_sr * 3,  # standard 3s clip
            truncation=True
        )

        return {
            "input_values": inputs.input_values.squeeze(0),
            "labels": torch.tensor(self.labels[idx])
        }


# -------------------- TRAINING --------------------
def train_system(train_files, train_labels, val_files, val_labels, epochs=5):
    """
    Fine-tune Wav2Vec2 model on custom emotion dataset.
    """
    model_name = "facebook/wav2vec2-base"

    processor = Wav2Vec2Processor.from_pretrained(model_name)
    model = Wav2Vec2ForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(EMOTION_LABELS)
    )

    train_dataset = EmotionDataset(train_files, train_labels, processor)
    val_dataset = EmotionDataset(val_files, val_labels, processor)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=8)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for batch in train_loader:
            optimizer.zero_grad()

            outputs = model(
                batch["input_values"].to(device),
                labels=batch["labels"].to(device)
            )

            loss = outputs.loss
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1} | Loss: {total_loss / len(train_loader):.4f}")

        # Validation
        model.eval()
        correct, total = 0, 0

        with torch.no_grad():
            for batch in val_loader:
                outputs = model(batch["input_values"].to(device))
                preds = torch.argmax(outputs.logits, dim=-1)

                labels = batch["labels"].to(device)

                correct += (preds == labels).sum().item()
                total += labels.size(0)

        print(f"Validation Accuracy: {(correct / total) * 100:.2f}%")

    model.save_pretrained("./fine_tuned_emotion_model")
    processor.save_pretrained("./fine_tuned_emotion_model")