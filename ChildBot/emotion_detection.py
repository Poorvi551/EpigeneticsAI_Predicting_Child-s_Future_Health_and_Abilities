import os
import random
from typing import Dict, List, Tuple
import numpy as np
import json

try:
    # TensorFlow/Keras is required for the CNN model
    import tensorflow as tf
    from tensorflow.keras.models import load_model
    from tensorflow.keras.preprocessing.text import Tokenizer
    from tensorflow.keras.utils import pad_sequences
    from tensorflow.keras import layers, models, optimizers
    import sys
    print("Python executable:", sys.executable)
except Exception as exc:
    raise RuntimeError(
        "TensorFlow is required to run this script. Install with: pip install tensorflow"
    ) from exc


def build_sample_intents() -> Dict[str, List[str]]:
    """
    Returns a small, hardcoded intents knowledge base for emotions.
    Each intent maps to a list of example utterances.
    """
    return {
        "happy": [
            "I am feeling great today",
            "This is awesome",
            "I'm so happy",
            "What a wonderful day",
            "Feeling joyful and excited",
            "Smiling and cheerful",
            "Life is good",
        ],
        "sad": [
            "I feel down",
            "This makes me cry",
            "I am so sad",
            "Feeling blue",
            "Today is a gloomy day",
            "I am heartbroken",
            "Lonely and upset",
        ],
        "angry": [
            "I am furious",
            "This makes me angry",
            "I'm pissed off",
            "So mad right now",
            "I can't stand this",
            "Rage is building",
            "This is infuriating",
        ],
        "fear": [
            "I'm scared",
            "This is terrifying",
            "I feel afraid",
            "I am anxious",
            "This worries me",
            "I fear the worst",
            "Feeling nervous",
        ],
        "surprise": [
            "Wow that's unexpected",
            "I didn't see that coming",
            "What a surprise",
            "Totally shocked",
            "That's unbelievable",
            "I'm astonished",
            "So surprising",
        ],
        "neutral": [
            "I don't know",
            "Okay",
            "It is what it is",
            "Fine",
            "Nothing special",
            "Just normal",
            "Meh",
        ],
    }


def build_label_message_map() -> Dict[str, str]:
    """Maps each emotion to the message to print on prediction."""
    return {
        "happy": "Looking so much happy today!!",
        "sad": "It's okay to feel sad. I'm here for you.",
        "angry": "Take a deep breath. Let's calm down together.",
        "fear": "It's natural to feel fear. You're not alone.",
        "surprise": "Whoa! That was surprising!",
        "neutral": "Noted. You seem neutral.",
    }


def load_knowledge_base(
    path: str = "knowledge_base.json",
) -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
    """
    Loads a knowledge base JSON with structure containing an array 'questions'.
    Each item can have: 'question' (str), 'similar' (list[str]), 'answers' (list[str]), 'intents' (list[str]).

    Returns:
      - intents_examples: dict[label -> list of training examples]
      - label_to_answers: dict[label -> list of answer strings]

    If the file is missing or invalid, returns empty dicts.
    """
    if not os.path.exists(path):
        return {}, {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return {}, {}

    items = data.get("questions", []) if isinstance(data, dict) else []
    intents_examples: Dict[str, List[str]] = {}
    label_to_answers: Dict[str, List[str]] = {}

    for item in items:
        if not isinstance(item, dict):
            continue
        question_text = item.get("question")
        similar = item.get("similar") or []
        answers = item.get("answers") or []
        intents = item.get("intents") or []

        texts: List[str] = []
        if isinstance(question_text, str) and question_text.strip():
            texts.append(question_text.strip())
        if isinstance(similar, list):
            for s in similar:
                if isinstance(s, str) and s.strip():
                    texts.append(s.strip())

        if not intents:
            # If no intents provided, skip this item
            continue

        for label in intents:
            if not isinstance(label, str) or not label.strip():
                continue
            label = label.strip()
            if texts:
                intents_examples.setdefault(label, []).extend(texts)
            if answers:
                # keep all answers per label; used later when printing a response
                label_to_answers.setdefault(label, [])
                for a in answers:
                    if isinstance(a, str) and a.strip():
                        label_to_answers[label].append(a.strip())

    # Deduplicate examples/answers per label
    for d in (intents_examples, label_to_answers):
        for k, v in list(d.items()):
            seen: Dict[str, bool] = {}
            deduped: List[str] = []
            for s in v:
                if s not in seen:
                    seen[s] = True
                    deduped.append(s)
            d[k] = deduped

    return intents_examples, label_to_answers


def prepare_dataset(
    intents: Dict[str, List[str]],
    validation_fraction: float = 0.2,
    seed: int = 7,
) -> Tuple[List[str], List[int], List[str], List[int], Dict[int, str], Dict[str, int]]:
    """
    Flattens intents to texts and labels, and performs a deterministic split.
    Returns train_texts, train_labels, val_texts, val_labels, id2label, label2id
    """
    random.seed(seed)
    labels_sorted = sorted(intents.keys())
    label2id = {label: idx for idx, label in enumerate(labels_sorted)}
    id2label = {idx: label for label, idx in label2id.items()}

    texts: List[str] = []
    labels: List[int] = []
    for label, examples in intents.items():
        for text in examples:
            texts.append(text)
            labels.append(label2id[label])

    # Shuffle paired
    indices = list(range(len(texts)))
    random.shuffle(indices)
    texts = [texts[i] for i in indices]
    labels = [labels[i] for i in indices]

    split_index = max(1, int((1.0 - validation_fraction) * len(texts)))
    train_texts = texts[:split_index]
    train_labels = labels[:split_index]
    val_texts = texts[split_index:]
    val_labels = labels[split_index:]
    return train_texts, train_labels, val_texts, val_labels, id2label, label2id


def vectorize_texts(
    train_texts: List[str],
    val_texts: List[str],
    vocab_size: int = 5000,
    max_length: int = 20,
) -> Tuple[Tokenizer, np.ndarray, np.ndarray]:
    """Fits a Keras Tokenizer and returns padded train/val sequences."""
    tokenizer = Tokenizer(num_words=vocab_size, oov_token="<OOV>")
    tokenizer.fit_on_texts(train_texts)
    x_train = tokenizer.texts_to_sequences(train_texts)
    x_val = tokenizer.texts_to_sequences(val_texts)
    x_train = pad_sequences(x_train, maxlen=max_length, padding="post", truncating="post")
    x_val = pad_sequences(x_val, maxlen=max_length, padding="post", truncating="post")
    return tokenizer, x_train, x_val


def build_text_cnn(
    vocab_size: int,
    embedding_dim: int,
    max_length: int,
    num_classes: int,
    conv_filters: int = 64,
    kernel_sizes: Tuple[int, int, int] = (3, 4, 5),
    dropout_rate: float = 0.3,
) -> models.Model:
    """
    A simple Text CNN: Embedding -> [Conv1D+GlobalMaxPool]*k -> concat -> Dense classifier
    """
    inputs = layers.Input(shape=(max_length,), dtype="int32")
    x = layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_length)(inputs)

    conv_pools = []
    for k in kernel_sizes:
        c = layers.Conv1D(filters=conv_filters, kernel_size=k, activation="relu")(x)
        p = layers.GlobalMaxPooling1D()(c)
        conv_pools.append(p)

    x = layers.Concatenate()(conv_pools) if len(conv_pools) > 1 else conv_pools[0]
    x = layers.Dropout(dropout_rate)(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(dropout_rate)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs=inputs, outputs=outputs)
    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer=optimizers.Adam(learning_rate=3e-4),
        metrics=["accuracy"],
    )
    return model


class EmotionDetector:
    def __init__(
        self,
        intents: Dict[str, List[str]],
        vocab_size: int = 5000,
        max_length: int = 20,
        embedding_dim: int = 50,
        epochs: int = 30,
        batch_size: int = 8,
        seed: int = 7,
    ) -> None:
        self.intents = intents
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.embedding_dim = embedding_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.seed = seed

        (
            train_texts,
            train_labels,
            val_texts,
            val_labels,
            self.id2label,
            self.label2id,
        ) = prepare_dataset(intents, validation_fraction=0.25, seed=seed)

        self.tokenizer, x_train, x_val = vectorize_texts(
            train_texts, val_texts, vocab_size=vocab_size, max_length=max_length
        )
        self.y_train = np.array(train_labels, dtype=np.int32)
        self.y_val = np.array(val_labels, dtype=np.int32)

        self.model = build_text_cnn(
            vocab_size=min(vocab_size, len(self.tokenizer.word_index) + 1),
            embedding_dim=embedding_dim,
            max_length=max_length,
            num_classes=len(self.id2label),
        )

    def fit(self) -> None:
        self.model.fit(
            x=self._texts_to_padded(self._get_texts(self.y_train.size, split="train")),
            y=self.y_train,
            validation_data=(
                self._texts_to_padded(self._get_texts(self.y_val.size, split="val")),
                self.y_val,
            ),
            epochs=self.epochs,
            batch_size=self.batch_size,
            verbose=0,
        )

    def _get_texts(self, n: int, split: str) -> List[str]:
        # Helper to reconstruct texts corresponding to train/val given shapes
        # We store them again to avoid keeping large copies; this dataset is small.
        # Instead, we regenerate from tokenizer's index_docs order via simple slices.
        # For a real project, keep explicit arrays.
        all_texts: List[str] = []
        for label, examples in self.intents.items():
            for text in examples:
                all_texts.append(text)
        # Deterministic reordering to match prepare_dataset shuffle/split
        rnd = random.Random(self.seed)
        idx = list(range(len(all_texts)))
        rnd.shuffle(idx)
        texts_shuffled = [all_texts[i] for i in idx]
        split_index = max(1, int((1.0 - 0.25) * len(texts_shuffled)))
        if split == "train":
            return texts_shuffled[:split_index]
        return texts_shuffled[split_index:]

    def _texts_to_padded(self, texts: List[str]) -> np.ndarray:
        seqs = self.tokenizer.texts_to_sequences(texts)
        return pad_sequences(seqs, maxlen=self.max_length, padding="post", truncating="post")

    def predict(self, text: str) -> Tuple[str, float]:
        seq = self._texts_to_padded([text])
        probs = self.model.predict(seq, verbose=0)[0]
        idx = int(np.argmax(probs))
        label = self.id2label[idx]
        confidence = float(probs[idx])
        return label, confidence


def predict_emotion_and_print(text: str, detector: EmotionDetector, kb_label_answers: Dict[str, List[str]] | None = None) -> str:
    label_to_message = build_label_message_map()
    label, confidence = detector.predict(text)
    # Prefer KB answer if available for the predicted label
    if kb_label_answers and kb_label_answers.get(label):
        message = random.choice(kb_label_answers[label])
    else:
        message = label_to_message.get(label, f"Detected emotion: {label}")
    output = f"{message} (confidence: {confidence:.2f})"
    print(output)
    return output


def build_kb_phrase_index(path: str = "knowledge_base.json") -> Dict[str, List[str]]:
    """Builds a simple phrase->answers map from KB for quick exact matching."""
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return {}

    items = data.get("questions", []) if isinstance(data, dict) else []
    phrase_to_answers: Dict[str, List[str]] = {}
    for item in items:
        if not isinstance(item, dict):
            continue
        answers = [a.strip() for a in (item.get("answers") or []) if isinstance(a, str) and a.strip()]
        if not answers:
            continue
        phrases: List[str] = []
        q = item.get("question")
        if isinstance(q, str) and q.strip():
            phrases.append(q.strip())
        for s in (item.get("similar") or []):
            if isinstance(s, str) and s.strip():
                phrases.append(s.strip())
        for p in phrases:
            phrase_to_answers[p.lower()] = answers
    return phrase_to_answers

def _import_cv_libs():
    try:
        import cv2  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "OpenCV not found. Install with: pip install opencv-python"
        ) from exc
    try:
        from fer import FER  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "FER library not found. Install with: pip install fer"
        ) from exc
    return cv2, FER


def map_fer_label_to_message(label: str) -> str:
    mapping = build_label_message_map()
    if label in mapping:
        return mapping[label]
    # Reasonable fallbacks for labels not in our small KB
    fallback_map = {
        "disgust": "That looked unpleasant. Hope you feel better soon.",
        "contempt": "Sensed some contempt.",
    }
    return fallback_map.get(label, f"Detected emotion: {label}")


def run_webcam_expression_detection(min_confidence: float = 0.5) -> None:
    cv2, FER = _import_cv_libs()
    detector = FER(mtcnn=False)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not access webcam (device 0)")

    print("Webcam expression detection started. Press 'q' to quit.")
    last_label = None
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            results = detector.detect_emotions(frame) or []
            top_label = None
            top_score = 0.0
            if results:
                # FER returns a list of dicts { 'box': ..., 'emotions': {label: score} }
                emotions = results[0].get("emotions", {})
                if emotions:
                    top_label = max(emotions, key=emotions.get)
                    top_score = float(emotions[top_label])

            # Draw and print when confident and changed
            if top_label and top_score >= min_confidence:
                message = map_fer_label_to_message(top_label)
                if last_label != top_label:
                    print(f"{message} (confidence: {top_score:.2f})")
                    last_label = top_label
                cv2.putText(
                    frame,
                    f"{top_label}: {top_score:.2f}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )

            cv2.imshow("Emotion Detection", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


def main() -> None:
    # Load KB-backed intents/answers if present
    kb_intents, kb_label_answers = load_knowledge_base()
    kb_phrase_to_answers = build_kb_phrase_index()
    intents = kb_intents if kb_intents else build_sample_intents()

    # Keep training very fast for demo; use environment variables to override
    epochs = int(os.environ.get("EMOTION_EPOCHS", "30"))
    embedding_dim = int(os.environ.get("EMOTION_EMBED_DIM", "50"))

    detector = EmotionDetector(
        intents=intents,
        epochs=epochs,
        embedding_dim=embedding_dim,
        batch_size=8,
        max_length=20,
        vocab_size=5000,
    )
    detector.fit()

    print("Select mode: 1) Text  2) Webcam")
    choice = input("> ").strip()
    if choice == "2":
        run_webcam_expression_detection()
        return

    print("Type a message to detect emotion (or 'quit' to exit):")
    while True:
        try:
            user_input = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break
        if user_input.lower() in {"quit", "exit"}:
            print("Bye!")
            break
        if not user_input:
            continue
        # 1) Try exact KB phrase match first
        if kb_phrase_to_answers:
            answers = kb_phrase_to_answers.get(user_input.lower())
            if answers:
                msg = random.choice(answers)
                print(f"{msg} (from KB)")
                continue
        # 2) Fall back to ML intent prediction and per-intent KB answers
        predict_emotion_and_print(user_input, detector, kb_label_answers if kb_intents else None)


if __name__ == "__main__":
    main()