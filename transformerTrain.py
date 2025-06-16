import os
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LayerNormalization, Dropout, MultiHeadAttention, GlobalAveragePooling1D
import tensorflow as tf

# Configuration
output_dir = "gesture_data"
# model_path = "gesture_transformer_model.h5"
model_path_keras = "gesture_transformer_model.keras"
label_path = "gesture_labels.pkl"
SEQUENCE_LENGTH = 30  # number of frames per sequence

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

            # Ensure enough frames for sequence
            if len(data) >= SEQUENCE_LENGTH:
                for i in range(len(data) - SEQUENCE_LENGTH + 1):
                    X.append(data[i:i + SEQUENCE_LENGTH])
                    y.append(label)

    X = np.array(X)
    y = np.array(y)

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    y_onehot = to_categorical(y_encoded)

    return X, y_onehot, le

# Step 2: Build Transformer model
def build_transformer_model(input_shape, num_classes, num_heads=4, ff_dim=128):
    inputs = Input(shape=input_shape)
    x = Dense(input_shape[-1])(inputs)

    attn_output = MultiHeadAttention(num_heads=num_heads, key_dim=input_shape[-1])(x, x)
    x = LayerNormalization(epsilon=1e-6)(x + attn_output)

    ff_output = Dense(ff_dim, activation="relu")(x)
    ff_output = Dense(input_shape[-1])(ff_output)
    x = LayerNormalization(epsilon=1e-6)(x + ff_output)

    x = GlobalAveragePooling1D()(x)
    x = Dropout(0.3)(x)
    outputs = Dense(num_classes, activation="softmax")(x)

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    return model

# Step 3: Train and save the model
def train():
    print("Loading data...")
    X, y, label_encoder = load_sequence_data()
    if len(X) == 0:
        print("❌ No training data found.")
        return

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("Building model...")
    model = build_transformer_model(input_shape=X.shape[1:], num_classes=y.shape[1])
    model.summary()

    print("Training model...")
    history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=25, batch_size=32)

    print("Evaluating model...")
    loss, acc = model.evaluate(X_test, y_test)
    print(f"✅ Test Accuracy: {acc:.2f}")

    print("Saving model and labels...")
    # model.save(model_path)
    model.save(model_path_keras)
    with open(label_path, "wb") as f:
        pickle.dump(label_encoder, f)

    print("🎉 Model training complete and saved!")

if __name__ == "__main__":
    train()
