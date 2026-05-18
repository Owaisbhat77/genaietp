import os
import json
import zipfile
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf

# ---------- Config ----------
DATA_DIR = "data"
MODEL_DIR = "exported_model/1"
KAGGLE_DATASET = "uciml/sms-spam-collection-dataset"  # small text classification dataset
TEXT_COL = "message"
LABEL_COL = "label"

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# ---------- Download dataset ----------
def download_kaggle_dataset():
    # Requires kaggle.json in ~/.kaggle/kaggle.json
    os.system(f"kaggle datasets download -d {KAGGLE_DATASET} -p {DATA_DIR} --force")
    for file in os.listdir(DATA_DIR):
        if file.endswith(".zip"):
            with zipfile.ZipFile(os.path.join(DATA_DIR, file), "r") as zip_ref:
                zip_ref.extractall(DATA_DIR)

download_kaggle_dataset()

# ---------- Load data ----------
csv_path = None
for f in os.listdir(DATA_DIR):
    if f.endswith(".csv"):
        csv_path = os.path.join(DATA_DIR, f)
        break

if not csv_path:
    raise FileNotFoundError("CSV file not found in data directory.")

df = pd.read_csv(csv_path, encoding="latin-1")
df = df[[LABEL_COL, TEXT_COL]].dropna()

# map labels
label_map = {label: i for i, label in enumerate(sorted(df[LABEL_COL].unique()))}
df["label_id"] = df[LABEL_COL].map(label_map)

# split
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df["label_id"])
train_df, val_df = train_test_split(train_df, test_size=0.1, random_state=42, stratify=train_df["label_id"])

# TensorFlow datasets
def df_to_dataset(dframe, batch_size=32):
    return tf.data.Dataset.from_tensor_slices(
        (dframe[TEXT_COL].values, dframe["label_id"].values)
    ).batch(batch_size)

train_ds = df_to_dataset(train_df)
val_ds = df_to_dataset(val_df)
test_ds = df_to_dataset(test_df)

# Text vectorizer
vectorizer = tf.keras.layers.TextVectorization(max_tokens=10000, output_mode="int", output_sequence_length=100)
vectorizer.adapt(train_df[TEXT_COL].values)

# Model
inputs = tf.keras.Input(shape=(1,), dtype=tf.string)
x = vectorizer(inputs)
x = tf.keras.layers.Embedding(10000, 64)(x)
x = tf.keras.layers.GlobalAveragePooling1D()(x)
x = tf.keras.layers.Dense(64, activation="relu")(x)
outputs = tf.keras.layers.Dense(len(label_map), activation="softmax")(x)
model = tf.keras.Model(inputs, outputs)

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.summary()

model.fit(train_ds, validation_data=val_ds, epochs=3)

# Evaluate
loss, acc = model.evaluate(test_ds)
print(f"Test accuracy: {acc:.4f}")

# Save label map
os.makedirs("artifacts", exist_ok=True)
with open("artifacts/label_map.json", "w") as f:
    json.dump(label_map, f, indent=2)

# Export SavedModel for TF Serving
tf.saved_model.save(model, MODEL_DIR)
print(f"SavedModel exported to {MODEL_DIR}")
