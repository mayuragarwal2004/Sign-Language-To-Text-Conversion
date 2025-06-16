import os
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import MultiHeadAttention, LayerNormalization
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# Configuration
output_dir = "gesture_data"
model_path = "gesture_transformer_model.keras"
label_path = "gesture_labels.pkl"
SEQUENCE_LENGTH = 30

# Step 1: Load and preprocess data
def load_sequence_data():
    X = []
    y = []

    for label in os.listdir(output_dir):
        label_dir = os.path.join(output_dir, label)
        if not os.path.isdir(label_dir):
            continue

        for session in os.listdir(label_dir):
            session_dir = os.path.join(label_dir, session)
            data_file = os.path.join(session_dir, "normalized_coords.npy")
            if not os.path.exists(data_file):
                continue

            data = np.load(data_file)

            if len(data) >= SEQUENCE_LENGTH:
                for i in range(len(data) - SEQUENCE_LENGTH + 1):
                    X.append(data[i:i + SEQUENCE_LENGTH])
                    y.append(label)

    X = np.array(X)
    y = np.array(y)
    return X, y

# Step 2: Load model and label encoder
def test():
    print("Loading test data...")
    X, y = load_sequence_data()
    if len(X) == 0:
        print("❌ No data found.")
        return

    with open(label_path, "rb") as f:
        label_encoder = pickle.load(f)
    y_encoded = label_encoder.transform(y)
    y_onehot = to_categorical(y_encoded)

    X_train, X_test, y_train, y_test = train_test_split(X, y_onehot, test_size=0.2, random_state=42)

    print("Loading model with custom layers...")
    model = load_model(model_path, custom_objects={
        "MultiHeadAttention": MultiHeadAttention,
        "LayerNormalization": LayerNormalization
    })

    print("Evaluating on test set...")
    loss, acc = model.evaluate(X_test, y_test)
    print(f"✅ Test Accuracy: {acc:.2f}")

    # Optional: Predict a few samples
    predictions = model.predict(X_test[:5])
    predicted_labels = label_encoder.inverse_transform(np.argmax(predictions, axis=1))
    actual_labels = label_encoder.inverse_transform(np.argmax(y_test[:5], axis=1))

    print("\nSample predictions:")
    for i in range(5):
        print(f"👉 Predicted: {predicted_labels[i]} | Actual: {actual_labels[i]}")

if __name__ == "__main__":
    test()
