pip install tensorflow torch torchvision timm pennylane tensorflow-addons matplotlib scikit-learn pandas numpy

import os

# Base path to the data
data_path = "/kaggle/input/deepfake-and-real-images/Dataset"

# Print the directory structure
for root, dirs, files in os.walk(data_path):
    print(f"Directory: {root}")
    for file in files[:5]:  # Print only the first 5 files for brevity
        print(f"  File: {file}")
    print()

!pip install torch torchvision

import os
import tensorflow as tf
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import tensorflow as tf

try:
    # Check for available GPUs
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        # Use the first GPU if available
        strategy = tf.distribute.OneDeviceStrategy(device="/gpu:0")
        print("GPU initialized.")
    else:
        # Fallback to CPU if no GPUs are found
        strategy = tf.distribute.get_strategy()
        print("No GPU found, using default CPU strategy.")
except RuntimeError as e:
    # Handle potential errors during GPU initialization
    print(f"Error initializing GPU: {e}")
    strategy = tf.distribute.get_strategy() #for CPU
    print("Using default CPU strategy.")

print("REPLICAS: ", strategy.num_replicas_in_sync)

import tensorflow as tf
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os

# Define transforms with augmentation and normalization
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),               # Random horizontal flip
    transforms.RandomRotation(15),                   # Random rotation within 15 degrees
    transforms.Resize((224, 224)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),# Resize to 224x224
    transforms.ToTensor(),                           # Convert to tensor: shape (C, H, W)
    transforms.Normalize(mean=[0.485, 0.456, 0.406], # ImageNet normalization
                         std=[0.229, 0.224, 0.225]),
    transforms.Lambda(lambda x: x.permute(1, 2, 0))  # Permute to (H, W, C) for TensorFlow
])
transform_test = transforms.Compose([
    transforms.RandomHorizontalFlip(),               # Random horizontal flip
    #transforms.RandomRotation(25),                   # Random rotation within 15 degrees
    transforms.Resize((224, 224)),
    #transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    #transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),# Resize to 224x224
    transforms.ToTensor(),                           # Convert to tensor: shape (C, H, W)
    transforms.Normalize(mean=[0.485, 0.456, 0.406], # ImageNet normalization
                         std=[0.229, 0.224, 0.225]),
    transforms.Lambda(lambda x: x.permute(1, 2, 0))  # Permute to (H, W, C) for TensorFlow
])
# Set up dataset paths
data_dir = '/kaggle/input/deepfake-and-real-images/Dataset'
train_dataset = datasets.ImageFolder(os.path.join(data_dir, 'Train'), transform=transform)
val_dataset   = datasets.ImageFolder(os.path.join(data_dir, 'Validation'), transform=transform_test)
test_dataset  = datasets.ImageFolder(os.path.join(data_dir, 'Test'),transform=transform_test)

# Data loaders (PyTorch)
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

print(f"Number of training samples: {len(train_dataset)}")
print(f"Number of validation samples: {len(val_dataset)}")
print(f"Number of test samples: {len(test_dataset)}")

# Define a generator to yield batches from a DataLoader
def loader_generator(loader):
    for x, y in loader:
        # x: torch.Tensor of shape (N, H, W, C) after permute
        # Convert to NumPy array with dtype float32
        yield x.numpy().astype('float32'), y.numpy()

# Wrap PyTorch DataLoaders as tf.data.Datasets
output_signature = (
    tf.TensorSpec(shape=(None, 224, 224, 3), dtype=tf.float32),
    tf.TensorSpec(shape=(None,), dtype=tf.int64)
)

train_dataset_tf = tf.data.Dataset.from_generator(
    lambda: loader_generator(train_loader),
    output_signature=output_signature
).prefetch(tf.data.AUTOTUNE)

val_dataset_tf = tf.data.Dataset.from_generator(
    lambda: loader_generator(val_loader),
    output_signature=output_signature
).prefetch(tf.data.AUTOTUNE)

test_dataset_tf = tf.data.Dataset.from_generator(
    lambda: loader_generator(test_loader),
    output_signature=output_signature
).prefetch(tf.data.AUTOTUNE)

for x_batch, y_batch in train_dataset_tf.take(1):
    print("x_batch shape:", x_batch.shape)  # Expected: (16, 224, 224, 3)
    print("y_batch shape:", y_batch.shape)  # Expected: (16,)
    break

import tensorflow as tf
import numpy as np
import pennylane as qml

# Quantum circuit parameters
n_qubits = 4
n_layers = 2

# Define the quantum device
dev = qml.device("default.qubit", wires=n_qubits)

# Define the quantum circuit
def quantum_circuit(inputs, weights):
    # Rotate each qubit with the input angles
    for i in range(n_qubits):
        qml.RY(inputs[i], wires=i)
    # Apply parameterized rotations and entangling layers
    for layer in range(n_layers):
        for qubit in range(n_qubits):
            qml.RX(weights[layer, qubit, 0], wires=qubit)
            qml.RY(weights[layer, qubit, 1], wires=qubit)
            qml.RZ(weights[layer, qubit, 2], wires=qubit)
        # Entangle the qubits
        for qubit in range(n_qubits - 1):
            qml.CNOT(wires=[qubit, qubit + 1])
        qml.CNOT(wires=[n_qubits - 1, 0])
    # Return expectation values from each qubit
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

# Create the QNode (using TensorFlow interface)
qnode = qml.QNode(quantum_circuit, dev, interface="tf")

# Define a custom quantum layer using tf.keras.layers.Layer
class CustomQuantumLayer(tf.keras.layers.Layer):
    def __init__(self, n_qubits, n_layers, **kwargs):
        super(CustomQuantumLayer, self).__init__(**kwargs)
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.weight_shape = (n_layers, n_qubits, 3)
        # Create the projection layers once in the constructor
        self.flatten_layer = tf.keras.layers.Flatten()
        self.projection = tf.keras.layers.Dense(self.n_qubits, activation='tanh')

    def build(self, input_shape):
        # Create the trainable quantum weights once in build
        self.quantum_weights = self.add_weight(
            name='quantum_weights',
            shape=self.weight_shape,
            initializer=tf.random_uniform_initializer(0, 2 * np.pi),
            trainable=True
        )
        super(CustomQuantumLayer, self).build(input_shape)

    def call(self, inputs):
        # Use the pre-created flatten and projection layers
        x = self.flatten_layer(inputs)
        x = self.projection(x)
        x = tf.clip_by_value(x, -np.pi, np.pi)
        x = tf.cast(x, dtype=tf.float64)

        # Define function to apply the QNode for one sample
        def apply_qnode(x_single):
            # x_single has shape (n_qubits,)
            result = qnode(x_single, self.quantum_weights)
            return tf.convert_to_tensor(result, dtype=tf.float32)

        # Apply the qnode across the batch using tf.map_fn
        qnode_outputs = tf.map_fn(apply_qnode, x, fn_output_signature=tf.float32)
        # Ensure the output has static shape (batch_size, n_qubits)
        qnode_outputs = tf.ensure_shape(qnode_outputs, (None, self.n_qubits))
        return qnode_outputs

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.n_qubits)


# ---------------------------
# Assume that the following functions are defined:
# create_xception_backbone(input_shape) and create_vit_backbone(input_shape)
# (They should output tensors of shape (None, features).)
# For brevity, here's a recap of simplified versions:

from tensorflow.keras.applications import Xception
from tensorflow.keras.layers import GlobalAveragePooling2D, Input, Dense, Concatenate, Dropout, LayerNormalization
from tensorflow.keras.models import Model

def create_xception_backbone(input_shape):
    base_model = Xception(weights='imagenet', include_top=False, input_shape=input_shape)
    x = GlobalAveragePooling2D()(base_model.output)
    return Model(inputs=base_model.input, outputs=x)

# Updated ViT backbone that applies global average pooling to match shapes:
from transformers import ViTModel
import torch

from transformers import ViTModel
import torch

def create_vit_backbone(input_shape):
    # Load the pre-trained ViT model (ViT-base)
    vit_model = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
    vit_model.eval()
    inputs = tf.keras.Input(shape=input_shape)

    def vit_forward(x):
        # Convert TF tensor to NumPy and then to Torch tensor
        x = tf.convert_to_tensor(x)
        x = x.numpy()
        x = torch.tensor(x).permute(0, 3, 1, 2)  # from (B, H, W, C) to (B, C, H, W)
        x = x / 255.0  # Normalize to [0,1]
        with torch.no_grad():
            outputs = vit_model(x).last_hidden_state  # (B, seq_length, hidden_dim)
        return outputs.numpy()

    # Wrap with tf.py_function and specify output shape using a Lambda with tf.ensure_shape
    vit_outputs = tf.keras.layers.Lambda(
        lambda x: tf.ensure_shape(tf.py_function(func=vit_forward, inp=[x], Tout=tf.float32), (None, 197, 768))
    )(inputs)

    # Global average pooling to reduce to shape (None, 768)
    vit_outputs = tf.keras.layers.GlobalAveragePooling1D()(vit_outputs)
    return tf.keras.Model(inputs=inputs, outputs=vit_outputs)

# ---------------------------
# Now, build the hybrid model by merging the two backbones and adding the QNN layer

def build_hybrid_model(input_shape):
    inputs = Input(shape=input_shape)

    # Xception Backbone
    xception = create_xception_backbone(input_shape)
    xception_features = xception(inputs)  # Expected shape: (None, 2048)

    # ViT Backbone
    vit = create_vit_backbone(input_shape)
    vit_features = vit(inputs)  # Expected shape after pooling: (None, 768)

    # Optionally, project features to a common dimension before merging.
    # Here, we project the ViT features to 2048 to match Xception.
    vit_proj = Dense(2048, activation='relu')(vit_features)

    # Merge features
    merged_features = Concatenate()([xception_features, vit_proj])  # Shape: (None, 4096)

    # Optionally, reduce dimension before feeding to the QNN layer
    merged_features = Dense(128, activation='relu')(merged_features)
    merged_features = LayerNormalization(epsilon=1e-6)(merged_features)
    merged_features = Dropout(0.3)(merged_features)

    # QNN Layer
    qnn_out = CustomQuantumLayer(n_qubits=n_qubits, n_layers=n_layers)(merged_features)
    qnn_out = Dense(32, activation='relu')(qnn_out)

    # Merge QNN output with previous merged features
    combined = Concatenate()([merged_features, qnn_out])

    # Final classification layers
    x = Dense(64, activation='relu')(combined)
    x = Dropout(0.2)(x)
    outputs = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model

# Build the hybrid model
hybrid_model = build_hybrid_model((224, 224, 3))
hybrid_model.summary()

import tensorflow as tf
import numpy as np
import pennylane as qml
import matplotlib.pyplot as plt
import gc
from tensorflow.keras import layers, Model, Input
from tensorflow.keras.applications import Xception
from tensorflow.keras import mixed_precision
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import os
from sklearn.model_selection import train_test_split

# Set mixed precision policy
mixed_precision.set_global_policy('mixed_float16')

# Confirm GPU and versions
print("GPUs available:", tf.config.list_physical_devices('GPU'))
print("Pennylane version:", qml.__version__)
print("TensorFlow version:", tf.__version__)

# Quantum circuit parameters
n_qubits = 4
n_layers = 2

dev = qml.device("default.qubit", wires=n_qubits)

def quantum_circuit(inputs, weights):
    for i in range(n_qubits):
        qml.RY(inputs[i], wires=i)
    for layer in range(n_layers):
        for qubit in range(n_qubits):
            qml.RX(weights[layer, qubit, 0], wires=qubit)
            qml.RY(weights[layer, qubit, 1], wires=qubit)
            qml.RZ(weights[layer, qubit, 2], wires=qubit)
        for qubit in range(n_qubits - 1):
            qml.CNOT(wires=[qubit, qubit + 1])
        qml.CNOT(wires=[n_qubits - 1, 0])
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

qnode = qml.QNode(quantum_circuit, dev, interface="tf")

class CustomQuantumLayer(tf.keras.layers.Layer):
    def __init__(self, n_qubits, n_layers, **kwargs):
        super(CustomQuantumLayer, self).__init__(**kwargs)
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.weight_shape = (n_layers, n_qubits, 3)
        self.flatten_layer = tf.keras.layers.Flatten()
        self.projection = tf.keras.layers.Dense(self.n_qubits, activation='tanh')

    def build(self, input_shape):
        self.quantum_weights = self.add_weight(
            name='quantum_weights',
            shape=self.weight_shape,
            initializer=tf.random_uniform_initializer(0, 2 * np.pi),
            trainable=True,
            dtype=tf.float32
        )
        super(CustomQuantumLayer, self).build(input_shape)

    @tf.function
    def call(self, inputs):
        with tf.device('/GPU:0'):
            x = self.flatten_layer(inputs)
            x = self.projection(x)
            x = tf.clip_by_value(x, -np.pi, np.pi)
            x = tf.cast(x, dtype=tf.float32)

        @tf.function
        def apply_qnode(x_single):
            x_single = tf.ensure_shape(x_single, (self.n_qubits,))
            result = qnode(x_single, self.quantum_weights)
            return tf.cast(result, tf.float32)

        qnode_outputs = tf.map_fn(
            apply_qnode,
            x,
            fn_output_signature=tf.TensorSpec(shape=(self.n_qubits,), dtype=tf.float32)
        )
        qnode_outputs = tf.ensure_shape(qnode_outputs, (None, self.n_qubits))
        return qnode_outputs

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.n_qubits)

class PatchEmbedding(layers.Layer):
    def __init__(self, patch_size=16, embedding_dim=768):
        super(PatchEmbedding, self).__init__()
        self.patch_size = patch_size
        self.embedding_dim = embedding_dim
        self.projection = layers.Conv2D(
            filters=embedding_dim,
            kernel_size=patch_size,
            strides=patch_size,
            padding="VALID"
        )

    def call(self, x):
        x = self.projection(x)
        batch_size = tf.shape(x)[0]
        x = tf.reshape(x, [batch_size, -1, self.embedding_dim])
        return x

class PositionalEmbedding(layers.Layer):
    def __init__(self, num_patches, embedding_dim):
        super(PositionalEmbedding, self).__init__()
        self.pos_embedding = layers.Embedding(input_dim=num_patches + 1, output_dim=embedding_dim)
        self.class_token = self.add_weight(shape=(1, 1, embedding_dim), initializer="zeros", trainable=True)

    def call(self, x):
        batch_size = tf.shape(x)[0]
        cls_token = tf.repeat(self.class_token, batch_size, axis=0)
        x = tf.concat([cls_token, x], axis=1)
        positions = tf.range(start=0, limit=tf.shape(x)[1], delta=1)
        pos_embeddings = self.pos_embedding(positions)
        return x + pos_embeddings

class TransformerEncoder(layers.Layer):
    def __init__(self, embedding_dim, num_heads, mlp_dim, dropout=0.4):
        super(TransformerEncoder, self).__init__()
        self.layer_norm1 = layers.LayerNormalization(epsilon=1e-6)
        self.attention = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embedding_dim // num_heads, dropout=dropout)
        self.layer_norm2 = layers.LayerNormalization(epsilon=1e-6)
        self.mlp = tf.keras.Sequential([
            layers.Dense(mlp_dim, activation=tf.nn.gelu),
            layers.Dropout(dropout),
            layers.Dense(embedding_dim),
            layers.Dropout(dropout)
        ])

    def call(self, x, training=False):
        attn_output = self.attention(self.layer_norm1(x), self.layer_norm1(x), training=training)
        x = x + attn_output
        mlp_output = self.mlp(self.layer_norm2(x), training=training)
        return x + mlp_output

def create_tf_vit(input_shape, patch_size=16, embedding_dim=768, num_transformer_layers=3, num_heads=12, mlp_dim=2048, dropout=0.4):
    inputs = Input(shape=input_shape)
    h, w = input_shape[0], input_shape[1]
    num_patches = (h // patch_size) * (w // patch_size)
    x = PatchEmbedding(patch_size, embedding_dim)(inputs)
    x = PositionalEmbedding(num_patches, embedding_dim)(x)
    for _ in range(num_transformer_layers):
        x = TransformerEncoder(embedding_dim, num_heads, mlp_dim, dropout)(x)
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    x = x[:, 0]
    return Model(inputs, x)

def create_xception_backbone(input_shape):
    base_model = Xception(weights='imagenet', include_top=False, input_shape=input_shape)
    x = layers.GlobalAveragePooling2D()(base_model.output)
    return Model(inputs=base_model.input, outputs=x)

def build_hybrid_model(input_shape):
    inputs = Input(shape=input_shape)
    xception = create_xception_backbone(input_shape)
    xception_features = xception(inputs)

    vit = create_tf_vit(input_shape)
    vit_features = vit(inputs)

    vit_proj = layers.Dense(2048, activation='relu')(vit_features)
    merged_features = layers.Concatenate()([xception_features, vit_proj])

    merged_features = layers.Dense(128, activation='relu')(merged_features)
    merged_features = layers.LayerNormalization(epsilon=1e-6)(merged_features)
    merged_features = layers.Dropout(0.3)(merged_features)

    qnn_out = CustomQuantumLayer(n_qubits=n_qubits, n_layers=n_layers)(merged_features)
    qnn_out = layers.Dense(32, activation='relu')(qnn_out)

    combined = layers.Concatenate()([merged_features, qnn_out])
    x = layers.Dense(64, activation='relu')(combined)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)

    model = Model(inputs=inputs, outputs=outputs)
    return model

def train_model_with_chunks(train_dataset, val_dataset, chunk_size=32, val_chunk_size=32):
    strategy = tf.distribute.MirroredStrategy()
    print(f"Running on {len(tf.config.list_physical_devices('GPU'))} GPU(s)")
    print(f"Number of replicas: {strategy.num_replicas_in_sync}")

    input_shape = (224, 224, 3)
    epochs =  10
    patience = 5
    best_val_loss = float('inf')
    patience_counter = 0
    epoch_losses = []
    epoch_val_losses = []
    epoch_val_accs = []

    with strategy.scope():
        hybrid_model = build_hybrid_model(input_shape)
        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
        loss_fn = tf.keras.losses.BinaryCrossentropy()
        train_acc_metric = tf.keras.metrics.BinaryAccuracy()
        val_acc_metric = tf.keras.metrics.BinaryAccuracy()

        @tf.function
        def train_step(x, y):
            with tf.device('/GPU:0'):
                with tf.GradientTape() as tape:
                    logits = hybrid_model(x, training=True)
                    loss_value = loss_fn(y, logits)
                grads = tape.gradient(loss_value, hybrid_model.trainable_weights)
                optimizer.apply_gradients(zip(grads, hybrid_model.trainable_weights))
                train_acc_metric.update_state(y, logits)
                return loss_value

        dummy_batch = tf.zeros((16, 224, 224, 3), dtype=tf.float16)
        dummy_labels = tf.zeros((16,), dtype=tf.float32)
        strategy.run(train_step, args=(dummy_batch, dummy_labels))

    checkpoint_path = "best_hybrid_model.keras"

    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        if epoch == 4:
            print("Freezing Xception layers")
            xception_model = hybrid_model.get_layer(hybrid_model.layers[1].name)  # Assumes Xception is the second layer
            #for layer in xception_model.layers:
            for layer in xception_model.layers[:-5]:
                layer.trainable = False
            print(f"Trainable Xception layers: {[layer.name for layer in xception_model.layers[-5:]]}")

            with strategy.scope():
                hybrid_model.compile(optimizer=optimizer, loss=loss_fn, metrics=['accuracy'])
            print("Model summary after freezing Xception:")
            hybrid_model.summary()

        train_loss_total = 0
        train_batches = 0
        chunk_loss = 0
        batch_count = 0

        for batch_idx, (x_batch, y_batch) in enumerate(train_dataset):
            loss_value = strategy.run(train_step, args=(x_batch, y_batch))
            chunk_loss += loss_value.numpy()
            batch_count += 1

            if (batch_idx + 1) % chunk_size == 0:
                print(f"  Training chunk {batch_idx // chunk_size + 1} loss: {chunk_loss / batch_count:.4f}")
                train_loss_total += chunk_loss
                train_batches += batch_count
                chunk_loss = 0
                batch_count = 0
                gc.collect()

        if batch_count > 0:
            print(f"  Training chunk {(batch_idx // chunk_size) + 1} loss: {chunk_loss / batch_count:.4f}")
            train_loss_total += chunk_loss
            train_batches += batch_count

        train_loss = train_loss_total / max(1, train_batches)
        train_acc = train_acc_metric.result().numpy()
        train_acc_metric.reset_state()

        val_loss_total = 0
        val_batches = 0
        val_chunk_loss = 0
        val_chunk_acc_total = 0
        val_batch_count = 0

        for val_batch_idx, (x_batch, y_batch) in enumerate(val_dataset):
            with tf.device('/GPU:0'):
                val_logits = hybrid_model(x_batch, training=False)
                val_loss = loss_fn(y_batch, val_logits)
            val_acc_metric.update_state(y_batch, val_logits)
            val_chunk_loss += val_loss.numpy()
            val_chunk_acc_total += val_acc_metric.result().numpy()
            val_acc_metric.reset_state()
            val_batch_count += 1

            if (val_batch_idx + 1) % val_chunk_size == 0:
                chunk_avg_loss = val_chunk_loss / val_batch_count
                chunk_avg_acc = val_chunk_acc_total / val_batch_count
                print(f"  Validation chunk {val_batch_idx // val_chunk_size + 1} loss: {chunk_avg_loss:.4f}, accuracy: {chunk_avg_acc:.4f}")
                val_loss_total += val_chunk_loss
                val_batches += val_batch_count
                val_chunk_loss = 0
                val_chunk_acc_total = 0
                val_batch_count = 0
                gc.collect()

        if val_batch_count > 0:
            chunk_avg_loss = val_chunk_loss / val_batch_count
            chunk_avg_acc = val_chunk_acc_total / val_batch_count
            print(f"  Validation chunk {(val_batch_idx // val_chunk_size) + 1} loss: {chunk_avg_loss:.4f}, accuracy: {chunk_avg_acc:.4f}")
            val_loss_total += val_chunk_loss
            val_batches += val_batch_count

        val_loss = val_loss_total / max(1, val_batches)
        val_acc = val_acc_metric.result().numpy()
        val_acc_metric.reset_state()

        epoch_losses.append(train_loss)
        epoch_val_losses.append(val_loss)
        epoch_val_accs.append(val_acc)

        print(f"Epoch {epoch+1} - loss: {train_loss:.4f}, accuracy: {train_acc:.4f}, val_loss: {val_loss:.4f}, val_accuracy: {val_acc:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            hybrid_model.save(checkpoint_path)
            print(f"Model saved to {checkpoint_path}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break

        gc.collect()

    best_model = tf.keras.models.load_model(
        checkpoint_path,
        custom_objects={'CustomQuantumLayer': CustomQuantumLayer}
    )

    plt.figure(figsize=(10, 6))
    plt.plot(epoch_losses, label='Training Loss', marker='x')
    plt.plot(epoch_val_losses, label='Validation Loss', marker='o')
    plt.title("Training vs Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('training_loss.png')
    plt.show()

    return best_model


# Subset training to 20K (optional, keeping your latest full 140K setup unless specified)
indices = list(range(len(train_dataset)))
labels = [train_dataset.targets[i] for i in indices]
train_indices, _ = train_test_split(indices, train_size=40000, stratify=labels, random_state=42)
train_dataset = Subset(train_dataset, train_indices)

# Subset validation to 10K
val_indices = list(range(len(val_dataset)))
val_labels = [val_dataset.targets[i] for i in val_indices]
if len(val_dataset) < 10000:
    raise ValueError(f"Validation dataset has only {len(val_dataset)} images, less than 10K requested.")
val_indices_subset, _ = train_test_split(val_indices, train_size=8000, stratify=val_labels, random_state=42)
val_dataset = Subset(val_dataset, val_indices_subset)

batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

print(f"Number of training samples: {len(train_dataset)}")
print(f"Number of validation samples: {len(val_dataset)}")
print(f"Number of test samples: {len(test_dataset)}")

def loader_generator(loader):
    for x, y in loader:
        yield x.numpy().astype('float32'), y.numpy()

output_signature = (
    tf.TensorSpec(shape=(None, 224, 224, 3), dtype=tf.float32),
    tf.TensorSpec(shape=(None,), dtype=tf.int64)
)

train_dataset_tf = tf.data.Dataset.from_generator(
    lambda: loader_generator(train_loader),
    output_signature=output_signature
).prefetch(tf.data.AUTOTUNE)

val_dataset_tf = tf.data.Dataset.from_generator(
    lambda: loader_generator(val_loader),
    output_signature=output_signature
).prefetch(tf.data.AUTOTUNE)

test_dataset_tf = tf.data.Dataset.from_generator(
    lambda: loader_generator(test_loader),
    output_signature=output_signature
).prefetch(tf.data.AUTOTUNE)

# Debug shapes and run
for x_batch, y_batch in train_dataset_tf.take(1):
    print("x_batch shape:", x_batch.shape)
    print("y_batch shape:", y_batch.shape)
    break

tf.debugging.set_log_device_placement(True)

best_model = train_model_with_chunks(train_dataset_tf, val_dataset_tf, chunk_size=64, val_chunk_size=64)

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import os

# Confirm we're still using mixed precision (matches training setup)
print("Mixed precision policy:", tf.keras.mixed_precision.global_policy().name)

# Verify best_model is loaded
if best_model is None:
    raise ValueError("best_model is not defined. Ensure the training script completed successfully.")

# Check test_dataset_tf is ready
try:
    for x_batch, y_batch in test_dataset_tf.take(1):
        print("Test dataset batch shape:", x_batch.shape, y_batch.shape)
        break
except Exception as e:
    raise ValueError(f"Error accessing test_dataset_tf: {e}")

# Evaluation function
def evaluate_model(model, dataset):
    y_true = []
    y_pred = []

    print("Running evaluation on test set...")
    for x_batch, y_batch in dataset:
        # Run predictions on GPU for speed
        with tf.device('/GPU:0'):
            predictions = model(x_batch, training=False)
            predictions = tf.squeeze(predictions, axis=-1)  # Shape: (batch_size,)
            predictions_binary = tf.cast(predictions >= 0.5, tf.int64)  # Threshold at 0.5

        y_true.extend(y_batch.numpy())
        y_pred.extend(predictions_binary.numpy())

    return np.array(y_true), np.array(y_pred)

# Run evaluation using existing best_model and test_dataset_tf
y_true, y_pred = evaluate_model(best_model, test_dataset_tf)

# Calculate accuracy and F1-score
accuracy = accuracy_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)

print(f"Test Accuracy: {accuracy:.4f}")
print(f"Test F1-Score: {f1:.4f}")

# Generate and plot confusion matrix heatmap
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['Real', 'Fake'], yticklabels=['Real', 'Fake'])
plt.title('Confusion Matrix - Deepfake Detection Model')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.tight_layout()

# Save heatmap
heatmap_path = 'confusion_matrix_heatmap.png'
plt.savefig(heatmap_path)
print(f"Heatmap saved to {heatmap_path}")
plt.show()

# Save metrics to a file
metrics_path = 'test_metrics.txt'
with open(metrics_path, 'w') as f:
    f.write(f"Test Accuracy: {accuracy:.4f}\n")
    f.write(f"Test F1-Score: {f1:.4f}\n")
print(f"Metrics saved to {metrics_path}")

import tensorflow as tf
import numpy as np
import pennylane as qml
import matplotlib.pyplot as plt
import gc
from tensorflow.keras import layers, Model, Input
from tensorflow.keras.applications import Xception
from tensorflow.keras import mixed_precision
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import os
from sklearn.model_selection import train_test_split

# Set mixed precision policy
mixed_precision.set_global_policy('mixed_float16')

# Confirm GPU and versions
print("GPUs available:", tf.config.list_physical_devices('GPU'))
print("Pennylane version:", qml.__version__)
print("TensorFlow version:", tf.__version__)

# Quantum circuit parameters
n_qubits = 4
n_layers = 2

dev = qml.device("default.qubit", wires=n_qubits)

def quantum_circuit(inputs, weights):
    for i in range(n_qubits):
        qml.RY(inputs[i], wires=i)
    for layer in range(n_layers):
        for qubit in range(n_qubits):
            qml.RX(weights[layer, qubit, 0], wires=qubit)
            qml.RY(weights[layer, qubit, 1], wires=qubit)
            qml.RZ(weights[layer, qubit, 2], wires=qubit)
        for qubit in range(n_qubits - 1):
            qml.CNOT(wires=[qubit, qubit + 1])
        qml.CNOT(wires=[n_qubits - 1, 0])
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

qnode = qml.QNode(quantum_circuit, dev, interface="tf")

class CustomQuantumLayer(tf.keras.layers.Layer):
    def __init__(self, n_qubits, n_layers, **kwargs):
        super(CustomQuantumLayer, self).__init__(**kwargs)
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.weight_shape = (n_layers, n_qubits, 3)
        self.flatten_layer = tf.keras.layers.Flatten()
        self.projection = tf.keras.layers.Dense(self.n_qubits, activation='tanh')

    def build(self, input_shape):
        self.quantum_weights = self.add_weight(
            name='quantum_weights',
            shape=self.weight_shape,
            initializer=tf.random_uniform_initializer(0, 2 * np.pi),
            trainable=True,
            dtype=tf.float32
        )
        super(CustomQuantumLayer, self).build(input_shape)

    @tf.function
    def call(self, inputs):
        with tf.device('/GPU:0'):
            x = self.flatten_layer(inputs)
            x = self.projection(x)
            x = tf.clip_by_value(x, -np.pi, np.pi)
            x = tf.cast(x, dtype=tf.float32)

        @tf.function
        def apply_qnode(x_single):
            x_single = tf.ensure_shape(x_single, (self.n_qubits,))
            result = qnode(x_single, self.quantum_weights)
            return tf.cast(result, tf.float32)

        qnode_outputs = tf.map_fn(
            apply_qnode,
            x,
            fn_output_signature=tf.TensorSpec(shape=(self.n_qubits,), dtype=tf.float32)
        )
        qnode_outputs = tf.ensure_shape(qnode_outputs, (None, self.n_qubits))
        return qnode_outputs

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.n_qubits)

class PatchEmbedding(layers.Layer):
    def __init__(self, patch_size=16, embedding_dim=768, l2_regularization=0.01):
        super(PatchEmbedding, self).__init__()
        self.patch_size = patch_size
        self.embedding_dim = embedding_dim
        self.projection = layers.Conv2D(
            filters=embedding_dim,
            kernel_size=patch_size,
            strides=patch_size,
            padding="VALID",
            kernel_regularizer=tf.keras.regularizers.l2(l2_regularization)
        )

    def call(self, x):
        x = self.projection(x)
        batch_size = tf.shape(x)[0]
        x = tf.reshape(x, [batch_size, -1, self.embedding_dim])
        return x


class PositionalEmbedding(layers.Layer):
    def __init__(self, num_patches, embedding_dim, l2_regularization=0.01):
        super(PositionalEmbedding, self).__init__()
        self.pos_embedding = layers.Embedding(
            input_dim=num_patches + 1,
            output_dim=embedding_dim,
            embeddings_regularizer=tf.keras.regularizers.l2(l2_regularization)
        )
        self.class_token = self.add_weight(
            shape=(1, 1, embedding_dim),
            initializer="zeros",
            trainable=True,
            regularizer=tf.keras.regularizers.l2(l2_regularization)
        )

    def call(self, x):
        batch_size = tf.shape(x)[0]
        cls_token = tf.repeat(self.class_token, batch_size, axis=0)
        x = tf.concat([cls_token, x], axis=1)
        positions = tf.range(start=0, limit=tf.shape(x)[1], delta=1)
        pos_embeddings = self.pos_embedding(positions)
        return x + pos_embeddings


class TransformerEncoder(layers.Layer):
    def __init__(self, embedding_dim, num_heads, mlp_dim, dropout=0.4, l2_regularization=0.01):
        super(TransformerEncoder, self).__init__()
        self.layer_norm1 = layers.LayerNormalization(epsilon=1e-6)
        self.attention = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=embedding_dim // num_heads,
            dropout=dropout,
            kernel_regularizer=tf.keras.regularizers.l2(l2_regularization)
        )
        self.layer_norm2 = layers.LayerNormalization(epsilon=1e-6)
        self.mlp = tf.keras.Sequential([
            layers.Dense(
                mlp_dim,
                activation=tf.nn.gelu,
                kernel_regularizer=tf.keras.regularizers.l2(l2_regularization)
            ),
            layers.Dropout(dropout),
            layers.Dense(
                embedding_dim,
                kernel_regularizer=tf.keras.regularizers.l2(l2_regularization)
            ),
            layers.Dropout(dropout)
        ])

    def call(self, x, training=False):
        attn_output = self.attention(self.layer_norm1(x), self.layer_norm1(x), training=training)
        x = x + attn_output
        mlp_output = self.mlp(self.layer_norm2(x), training=training)
        return x + mlp_output


def create_tf_vit(input_shape, patch_size=16, embedding_dim=768, num_transformer_layers=3,
                 num_heads=12, mlp_dim=2048, dropout=0.4, l2_regularization=0.1):
    inputs = layers.Input(shape=input_shape)
    h, w = input_shape[0], input_shape[1]
    num_patches = (h // patch_size) * (w // patch_size)

    x = PatchEmbedding(patch_size, embedding_dim, l2_regularization)(inputs)
    x = PositionalEmbedding(num_patches, embedding_dim, l2_regularization)(x)

    for _ in range(num_transformer_layers):
        x = TransformerEncoder(embedding_dim, num_heads, mlp_dim, dropout, l2_regularization)(x)

    x = layers.LayerNormalization(epsilon=1e-6)(x)
    x = x[:, 0]

    # If you want to add an output layer with regularization:
    # x = layers.Dense(num_classes, kernel_regularizer=tf.keras.regularizers.l2(l2_regularization))(x)

    return tf.keras.Model(inputs, x)

def create_xception_backbone(input_shape):
    base_model = Xception(weights='imagenet', include_top=False, input_shape=input_shape)
    x = layers.GlobalAveragePooling2D()(base_model.output)
    return Model(inputs=base_model.input, outputs=x)

def build_hybrid_model(input_shape):
    inputs = Input(shape=input_shape)
    xception = create_xception_backbone(input_shape)
    xception_features = xception(inputs)

    vit = create_tf_vit(input_shape)
    vit_features = vit(inputs)

    vit_proj = layers.Dense(2048, activation='relu')(vit_features)
    merged_features = layers.Concatenate()([xception_features, vit_proj])

    merged_features = layers.Dense(128, activation='relu')(merged_features)
    merged_features = layers.LayerNormalization(epsilon=1e-6)(merged_features)
    merged_features = layers.Dropout(0.3)(merged_features)

    qnn_out = CustomQuantumLayer(n_qubits=n_qubits, n_layers=n_layers)(merged_features)
    qnn_out = layers.Dense(32, activation='relu')(qnn_out)

    combined = layers.Concatenate()([merged_features, qnn_out])
    x = layers.Dense(64, activation='relu')(combined)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)

    model = Model(inputs=inputs, outputs=outputs)
    return model

def train_model_with_chunks(train_dataset, val_dataset, test_dataset, chunk_size=32, val_chunk_size=32, test_chunk_size=32):
    strategy = tf.distribute.MirroredStrategy()
    print(f"Running on {len(tf.config.list_physical_devices('GPU'))} GPU(s)")
    print(f"Number of replicas: {strategy.num_replicas_in_sync}")

    input_shape = (224, 224, 3)
    epochs = 10
    patience = 5
    best_val_loss = float('inf')
    patience_counter = 0
    epoch_losses = []
    epoch_val_losses = []
    epoch_val_accs = []
    best_model = None

    with strategy.scope():
        hybrid_model = build_hybrid_model(input_shape)
        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4, clipnorm=1.0)
        loss_fn = tf.keras.losses.BinaryCrossentropy()
        train_acc_metric = tf.keras.metrics.BinaryAccuracy()
        val_acc_metric = tf.keras.metrics.BinaryAccuracy()

        @tf.function
        def train_step(x, y):
            with tf.device('/GPU:0'):
                with tf.GradientTape() as tape:
                    logits = hybrid_model(x, training=True)
                    loss_value = loss_fn(y, logits)
                grads = tape.gradient(loss_value, hybrid_model.trainable_weights)
                optimizer.apply_gradients(zip(grads, hybrid_model.trainable_weights))
                train_acc_metric.update_state(y, logits)
                return loss_value

        dummy_batch = tf.zeros((32, 224, 224, 3), dtype=tf.float16)
        dummy_labels = tf.zeros((32,), dtype=tf.float32)
        strategy.run(train_step, args=(dummy_batch, dummy_labels))

    checkpoint_path = "best_hybrid_model.weights.h5"

    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        if epoch == 4:
            print("Freezing Xception layers")
            xception_model = hybrid_model.get_layer(hybrid_model.layers[1].name)  # Assumes Xception is the second layer
            for layer in xception_model.layers[:-10]:
                layer.trainable = False
            print(f"Trainable Xception layers: {[layer.name for layer in xception_model.layers[-10:]]}")

            with strategy.scope():
                hybrid_model.compile(optimizer=optimizer, loss=loss_fn, metrics=['accuracy'])
            print("Model summary after freezing Xception:")
            hybrid_model.summary()

        # Reset metrics at the beginning of each epoch
        train_acc_metric.reset_state()
        val_acc_metric.reset_state()

        train_loss_total = 0
        train_batches = 0
        chunk_loss = 0
        batch_count = 0

        for batch_idx, (x_batch, y_batch) in enumerate(train_dataset):
            loss_value = strategy.run(train_step, args=(x_batch, y_batch))
            chunk_loss += loss_value.numpy()
            batch_count += 1

            if (batch_idx + 1) % chunk_size == 0:
                print(f"  Training chunk {batch_idx // chunk_size + 1} loss: {chunk_loss / batch_count:.4f}")
                train_loss_total += chunk_loss
                train_batches += batch_count
                chunk_loss = 0
                batch_count = 0
                gc.collect()

        if batch_count > 0:
            print(f"  Training chunk {(batch_idx // chunk_size) + 1} loss: {chunk_loss / batch_count:.4f}")
            train_loss_total += chunk_loss
            train_batches += batch_count

        train_loss = train_loss_total / max(1, train_batches)
        train_acc = train_acc_metric.result().numpy()

        # CHANGED: Only reset train metric after reading the value
        train_acc_metric.reset_state()

        # Validation phase
        val_loss_total = 0
        val_batches = 0

        # CHANGED: Process validation data in chunks but accumulate metrics
        for val_batch_idx, (x_batch, y_batch) in enumerate(val_dataset):
            with tf.device('/GPU:0'):
                val_logits = hybrid_model(x_batch, training=False)
                val_loss = loss_fn(y_batch, val_logits)

            # CHANGED: Update metric without resetting after each batch
            val_acc_metric.update_state(y_batch, val_logits)
            val_loss_total += val_loss.numpy()
            val_batches += 1

            if (val_batch_idx + 1) % val_chunk_size == 0:
                # CHANGED: Display current cumulative metrics for monitoring
                current_val_loss = val_loss_total / val_batches
                current_val_acc = val_acc_metric.result().numpy()
                print(f"  Validation progress - chunk {val_batch_idx // val_chunk_size + 1} cumulative loss: {current_val_loss:.4f}, accuracy: {current_val_acc:.4f}")
                gc.collect()

        # CHANGED: Calculate final validation metrics after all batches
        val_loss = val_loss_total / max(1, val_batches)
        val_acc = val_acc_metric.result().numpy()

        # CHANGED: Reset validation metric only after getting the final result
        val_acc_metric.reset_state()

        epoch_losses.append(train_loss)
        epoch_val_losses.append(val_loss)
        epoch_val_accs.append(val_acc)

        print(f"Epoch {epoch+1} - loss: {train_loss:.4f}, accuracy: {train_acc:.4f}, val_loss: {val_loss:.4f}, val_accuracy: {val_acc:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save model weights instead of full model
            hybrid_model.save_weights(checkpoint_path)
            print(f"Model weights saved to {checkpoint_path}")
            # Keep a reference to the current best model
            best_model = hybrid_model
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs")
                # In case of early stopping, load best weights
                try:
                    hybrid_model.load_weights(checkpoint_path)
                    best_model = hybrid_model
                    print("Loaded best model weights from checkpoint")
                except:
                    print("Could not load best weights, using current model")
                    best_model = hybrid_model
                break

        gc.collect()

    # If training completed without early stopping, make sure we use the best model
    if best_model is None or patience_counter >= patience:
        try:
            hybrid_model.load_weights(checkpoint_path)
            best_model = hybrid_model
            print("Loaded best model weights after training")
        except:
            print("Could not load best weights, using final model")
            best_model = hybrid_model

    # Plot training history
    plt.figure(figsize=(10, 6))
    plt.plot(epoch_losses, label='Training Loss', marker='x')
    plt.plot(epoch_val_losses, label='Validation Loss', marker='o')
    plt.title("Training vs Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('training_loss.png')
    plt.show()

    # Plot accuracy
    plt.figure(figsize=(10, 6))
    plt.plot(epoch_val_accs, label='Validation Accuracy', marker='o')
    plt.title("Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('validation_accuracy.png')
    plt.show()

    # CHANGED: Fixed evaluation on test dataset
    print("\nEvaluating on test dataset...")
    test_acc_metric = tf.keras.metrics.BinaryAccuracy()
    test_loss_total = 0
    test_batches = 0
    all_predictions = []
    all_labels = []

    for test_batch_idx, (x_batch, y_batch) in enumerate(test_dataset):
        with tf.device('/GPU:0'):
            test_logits = best_model(x_batch, training=False)
            test_loss = loss_fn(y_batch, test_logits)

        # Update the metric (don't reset after each batch)
        test_acc_metric.update_state(y_batch, test_logits)

        # Store predictions and labels for ROC curve
        all_predictions.extend(test_logits.numpy().flatten())
        all_labels.extend(y_batch.numpy().flatten())

        test_loss_total += test_loss.numpy()
        test_batches += 1

        if (test_batch_idx + 1) % test_chunk_size == 0:
            # CHANGED: Show progress with cumulative metrics
            current_test_loss = test_loss_total / test_batches
            current_test_acc = test_acc_metric.result().numpy()
            print(f"  Test progress - chunk {test_batch_idx // test_chunk_size + 1} cumulative loss: {current_test_loss:.4f}, accuracy: {current_test_acc:.4f}")
            gc.collect()

    # Final test metrics calculation after all batches
    test_loss = test_loss_total / max(1, test_batches)
    test_acc = test_acc_metric.result().numpy()

    print(f"Final test metrics - loss: {test_loss:.4f}, accuracy: {test_acc:.4f}")

    # Plot ROC curve
    from sklearn.metrics import roc_curve, auc
    fpr, tpr, _ = roc_curve(all_labels, all_predictions)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('roc_curve.png')
    plt.show()

    # Plot confusion matrix
    from sklearn.metrics import confusion_matrix
    import seaborn as sns

    # Convert predictions to binary
    binary_predictions = [1 if p >= 0.5 else 0 for p in all_predictions]
    cm = confusion_matrix(all_labels, binary_predictions)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    plt.show()

    return best_model

# # #Subset training to 20K (optional, keeping your latest full 140K setup unless specified)
indices = list(range(len(train_dataset)))
labels = [train_dataset.targets[i] for i in indices]
train_indices, _ = train_test_split(indices, train_size=40000, stratify=labels, random_state=42)
train_dataset = Subset(train_dataset, train_indices)

# Subset validation to 10K
val_indices = list(range(len(val_dataset)))
val_labels = [val_dataset.targets[i] for i in val_indices]
if len(val_dataset) < 10000:
    raise ValueError(f"Validation dataset has only {len(val_dataset)} images, less than 10K requested.")
val_indices_subset, _ = train_test_split(val_indices, train_size=10000, stratify=val_labels, random_state=42)
val_dataset = Subset(val_dataset, val_indices_subset)

batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

print(f"Number of training samples: {len(train_dataset)}")
print(f"Number of validation samples: {len(val_dataset)}")
print(f"Number of test samples: {len(test_dataset)}")

def loader_generator(loader):
    for x, y in loader:
        yield x.numpy().astype('float32'), y.numpy()

output_signature = (
    tf.TensorSpec(shape=(None, 224, 224, 3), dtype=tf.float32),
    tf.TensorSpec(shape=(None,), dtype=tf.int64)
)

train_dataset_tf = tf.data.Dataset.from_generator(
    lambda: loader_generator(train_loader),
    output_signature=output_signature
).prefetch(tf.data.AUTOTUNE)

val_dataset_tf = tf.data.Dataset.from_generator(
    lambda: loader_generator(val_loader),
    output_signature=output_signature
).prefetch(tf.data.AUTOTUNE)

test_dataset_tf = tf.data.Dataset.from_generator(
    lambda: loader_generator(test_loader),
    output_signature=output_signature
).prefetch(tf.data.AUTOTUNE)

# Debug shapes and run
for x_batch, y_batch in train_dataset_tf.take(1):
    print("x_batch shape:", x_batch.shape)
    print("y_batch shape:", y_batch.shape)
    break

tf.debugging.set_log_device_placement(True)

# Run training with test dataset evaluation
best_model = train_model_with_chunks(train_dataset_tf, val_dataset_tf, test_dataset_tf, chunk_size=64, val_chunk_size=64, test_chunk_size=64)

import tensorflow as tf
import numpy as np
import pennylane as qml
import matplotlib.pyplot as plt
import gc
from tensorflow.keras import layers, Model, Input
from tensorflow.keras.applications import Xception
from tensorflow.keras import mixed_precision
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import os
from sklearn.model_selection import train_test_split

# Set mixed precision policy
mixed_precision.set_global_policy('mixed_float16')

# Confirm GPU and versions
print("GPUs available:", tf.config.list_physical_devices('GPU'))
print("Pennylane version:", qml.__version__)
print("TensorFlow version:", tf.__version__)

# Quantum circuit parameters
n_qubits = 4
n_layers = 2

dev = qml.device("default.qubit", wires=n_qubits)

def quantum_circuit(inputs, weights):
    for i in range(n_qubits):
        qml.RY(inputs[i], wires=i)
    for layer in range(n_layers):
        for qubit in range(n_qubits):
            qml.RX(weights[layer, qubit, 0], wires=qubit)
            qml.RY(weights[layer, qubit, 1], wires=qubit)
            qml.RZ(weights[layer, qubit, 2], wires=qubit)
        for qubit in range(n_qubits - 1):
            qml.CNOT(wires=[qubit, qubit + 1])
        qml.CNOT(wires=[n_qubits - 1, 0])
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

qnode = qml.QNode(quantum_circuit, dev, interface="tf")

class CustomQuantumLayer(tf.keras.layers.Layer):
    def __init__(self, n_qubits, n_layers, **kwargs):
        super(CustomQuantumLayer, self).__init__(**kwargs)
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.weight_shape = (n_layers, n_qubits, 3)
        self.flatten_layer = tf.keras.layers.Flatten()
        self.projection = tf.keras.layers.Dense(self.n_qubits, activation='tanh')

    def build(self, input_shape):
        self.quantum_weights = self.add_weight(
            name='quantum_weights',
            shape=self.weight_shape,
            initializer=tf.random_uniform_initializer(0, 2 * np.pi),
            trainable=True,
            dtype=tf.float32
        )
        super(CustomQuantumLayer, self).build(input_shape)

    @tf.function
    def call(self, inputs):
        with tf.device('/GPU:0'):
            x = self.flatten_layer(inputs)
            x = self.projection(x)
            x = tf.clip_by_value(x, -np.pi, np.pi)
            x = tf.cast(x, dtype=tf.float32)

        @tf.function
        def apply_qnode(x_single):
            x_single = tf.ensure_shape(x_single, (self.n_qubits,))
            result = qnode(x_single, self.quantum_weights)
            return tf.cast(result, tf.float32)

        qnode_outputs = tf.map_fn(
            apply_qnode,
            x,
            fn_output_signature=tf.TensorSpec(shape=(self.n_qubits,), dtype=tf.float32)
        )
        qnode_outputs = tf.ensure_shape(qnode_outputs, (None, self.n_qubits))
        return qnode_outputs

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.n_qubits)

class PatchEmbedding(layers.Layer):
    def __init__(self, patch_size=16, embedding_dim=768, l2_regularization=0.01):
        super(PatchEmbedding, self).__init__()
        self.patch_size = patch_size
        self.embedding_dim = embedding_dim
        self.projection = layers.Conv2D(
            filters=embedding_dim,
            kernel_size=patch_size,
            strides=patch_size,
            padding="VALID",
            kernel_regularizer=tf.keras.regularizers.l2(l2_regularization)
        )

    def call(self, x):
        x = self.projection(x)
        batch_size = tf.shape(x)[0]
        x = tf.reshape(x, [batch_size, -1, self.embedding_dim])
        return x


class PositionalEmbedding(layers.Layer):
    def __init__(self, num_patches, embedding_dim, l2_regularization=0.01):
        super(PositionalEmbedding, self).__init__()
        self.pos_embedding = layers.Embedding(
            input_dim=num_patches + 1,
            output_dim=embedding_dim,
            embeddings_regularizer=tf.keras.regularizers.l2(l2_regularization)
        )
        self.class_token = self.add_weight(
            shape=(1, 1, embedding_dim),
            initializer="zeros",
            trainable=True,
            regularizer=tf.keras.regularizers.l2(l2_regularization)
        )

    def call(self, x):
        batch_size = tf.shape(x)[0]
        cls_token = tf.repeat(self.class_token, batch_size, axis=0)
        x = tf.concat([cls_token, x], axis=1)
        positions = tf.range(start=0, limit=tf.shape(x)[1], delta=1)
        pos_embeddings = self.pos_embedding(positions)
        return x + pos_embeddings


class TransformerEncoder(layers.Layer):
    def __init__(self, embedding_dim, num_heads, mlp_dim, dropout=0.4, l2_regularization=0.01):
        super(TransformerEncoder, self).__init__()
        self.layer_norm1 = layers.LayerNormalization(epsilon=1e-6)
        self.attention = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=embedding_dim // num_heads,
            dropout=dropout,
            kernel_regularizer=tf.keras.regularizers.l2(l2_regularization)
        )
        self.layer_norm2 = layers.LayerNormalization(epsilon=1e-6)
        self.mlp = tf.keras.Sequential([
            layers.Dense(
                mlp_dim,
                activation=tf.nn.gelu,
                kernel_regularizer=tf.keras.regularizers.l2(l2_regularization)
            ),
            layers.Dropout(dropout),
            layers.Dense(
                embedding_dim,
                kernel_regularizer=tf.keras.regularizers.l2(l2_regularization)
            ),
            layers.Dropout(dropout)
        ])

    def call(self, x, training=False):
        attn_output = self.attention(self.layer_norm1(x), self.layer_norm1(x), training=training)
        x = x + attn_output
        mlp_output = self.mlp(self.layer_norm2(x), training=training)
        return x + mlp_output


def create_tf_vit(input_shape, patch_size=16, embedding_dim=768, num_transformer_layers=3,
                 num_heads=12, mlp_dim=2048, dropout=0.4, l2_regularization=0.1):
    inputs = layers.Input(shape=input_shape)
    h, w = input_shape[0], input_shape[1]
    num_patches = (h // patch_size) * (w // patch_size)

    x = PatchEmbedding(patch_size, embedding_dim, l2_regularization)(inputs)
    x = PositionalEmbedding(num_patches, embedding_dim, l2_regularization)(x)

    for _ in range(num_transformer_layers):
        x = TransformerEncoder(embedding_dim, num_heads, mlp_dim, dropout, l2_regularization)(x)

    x = layers.LayerNormalization(epsilon=1e-6)(x)
    x = x[:, 0]

    # If you want to add an output layer with regularization:
    # x = layers.Dense(num_classes, kernel_regularizer=tf.keras.regularizers.l2(l2_regularization))(x)

    return tf.keras.Model(inputs, x)

def create_xception_backbone(input_shape):
    base_model = Xception(weights='imagenet', include_top=False, input_shape=input_shape)
    x = layers.GlobalAveragePooling2D()(base_model.output)
    return Model(inputs=base_model.input, outputs=x)

def build_hybrid_model(input_shape):
    inputs = Input(shape=input_shape)
    xception = create_xception_backbone(input_shape)
    xception_features = xception(inputs)

    vit = create_tf_vit(input_shape)
    vit_features = vit(inputs)

    vit_proj = layers.Dense(2048, activation='relu')(vit_features)
    merged_features = layers.Concatenate()([xception_features, vit_proj])

    merged_features = layers.Dense(128, activation='relu')(merged_features)
    merged_features = layers.LayerNormalization(epsilon=1e-6)(merged_features)
    merged_features = layers.Dropout(0.3)(merged_features)

    qnn_out = CustomQuantumLayer(n_qubits=n_qubits, n_layers=n_layers)(merged_features)
    qnn_out = layers.Dense(32, activation='relu')(qnn_out)

    combined = layers.Concatenate()([merged_features, qnn_out])
    x = layers.Dense(64, activation='relu')(combined)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)

    model = Model(inputs=inputs, outputs=outputs)
    return model

def train_model_with_chunks(train_dataset, val_dataset, test_dataset, chunk_size=32, val_chunk_size=32, test_chunk_size=32):
    strategy = tf.distribute.MirroredStrategy()
    print(f"Running on {len(tf.config.list_physical_devices('GPU'))} GPU(s)")
    print(f"Number of replicas: {strategy.num_replicas_in_sync}")

    input_shape = (224, 224, 3)
    epochs = 10
    patience = 5
    best_val_loss = float('inf')
    patience_counter = 0
    epoch_losses = []
    epoch_val_losses = []
    epoch_val_accs = []
    best_model = None

    with strategy.scope():
        hybrid_model = build_hybrid_model(input_shape)
        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4, clipnorm=1.0)
        loss_fn = tf.keras.losses.BinaryCrossentropy()
        train_acc_metric = tf.keras.metrics.BinaryAccuracy()
        val_acc_metric = tf.keras.metrics.BinaryAccuracy()

        @tf.function
        def train_step(x, y):
            with tf.device('/GPU:0'):
                with tf.GradientTape() as tape:
                    logits = hybrid_model(x, training=True)
                    loss_value = loss_fn(y, logits)
                grads = tape.gradient(loss_value, hybrid_model.trainable_weights)
                optimizer.apply_gradients(zip(grads, hybrid_model.trainable_weights))
                train_acc_metric.update_state(y, logits)
                return loss_value

        dummy_batch = tf.zeros((32, 224, 224, 3), dtype=tf.float16)
        dummy_labels = tf.zeros((32,), dtype=tf.float32)
        strategy.run(train_step, args=(dummy_batch, dummy_labels))

    checkpoint_path = "best_hybrid_model.weights.h5"

    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        if epoch == 4:
            print("Freezing Xception layers")
            xception_model = hybrid_model.get_layer(hybrid_model.layers[1].name)  # Assumes Xception is the second layer
            for layer in xception_model.layers[:-10]:
                layer.trainable = False
            print(f"Trainable Xception layers: {[layer.name for layer in xception_model.layers[-10:]]}")

            with strategy.scope():
                hybrid_model.compile(optimizer=optimizer, loss=loss_fn, metrics=['accuracy'])
            print("Model summary after freezing Xception:")
            hybrid_model.summary()

        # Reset metrics at the beginning of each epoch
        train_acc_metric.reset_state()
        val_acc_metric.reset_state()

        train_loss_total = 0
        train_batches = 0
        chunk_loss = 0
        batch_count = 0

        for batch_idx, (x_batch, y_batch) in enumerate(train_dataset):
            loss_value = strategy.run(train_step, args=(x_batch, y_batch))
            chunk_loss += loss_value.numpy()
            batch_count += 1

            if (batch_idx + 1) % chunk_size == 0:
                print(f"  Training chunk {batch_idx // chunk_size + 1} loss: {chunk_loss / batch_count:.4f}")
                train_loss_total += chunk_loss
                train_batches += batch_count
                chunk_loss = 0
                batch_count = 0
                gc.collect()

        if batch_count > 0:
            print(f"  Training chunk {(batch_idx // chunk_size) + 1} loss: {chunk_loss / batch_count:.4f}")
            train_loss_total += chunk_loss
            train_batches += batch_count

        train_loss = train_loss_total / max(1, train_batches)
        train_acc = train_acc_metric.result().numpy()

        # CHANGED: Only reset train metric after reading the value
        train_acc_metric.reset_state()

        # Validation phase
        val_loss_total = 0
        val_batches = 0

        # CHANGED: Process validation data in chunks but accumulate metrics
        for val_batch_idx, (x_batch, y_batch) in enumerate(val_dataset):
            with tf.device('/GPU:0'):
                val_logits = hybrid_model(x_batch, training=False)
                val_loss = loss_fn(y_batch, val_logits)

            # CHANGED: Update metric without resetting after each batch
            val_acc_metric.update_state(y_batch, val_logits)
            val_loss_total += val_loss.numpy()
            val_batches += 1

            if (val_batch_idx + 1) % val_chunk_size == 0:
                # CHANGED: Display current cumulative metrics for monitoring
                current_val_loss = val_loss_total / val_batches
                current_val_acc = val_acc_metric.result().numpy()
                print(f"  Validation progress - chunk {val_batch_idx // val_chunk_size + 1} cumulative loss: {current_val_loss:.4f}, accuracy: {current_val_acc:.4f}")
                gc.collect()

        # CHANGED: Calculate final validation metrics after all batches
        val_loss = val_loss_total / max(1, val_batches)
        val_acc = val_acc_metric.result().numpy()

        # CHANGED: Reset validation metric only after getting the final result
        val_acc_metric.reset_state()

        epoch_losses.append(train_loss)
        epoch_val_losses.append(val_loss)
        epoch_val_accs.append(val_acc)

        print(f"Epoch {epoch+1} - loss: {train_loss:.4f}, accuracy: {train_acc:.4f}, val_loss: {val_loss:.4f}, val_accuracy: {val_acc:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save model weights instead of full model
            hybrid_model.save_weights(checkpoint_path)
            print(f"Model weights saved to {checkpoint_path}")
            # Keep a reference to the current best model
            best_model = hybrid_model
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs")
                # In case of early stopping, load best weights
                try:
                    hybrid_model.load_weights(checkpoint_path)
                    best_model = hybrid_model
                    print("Loaded best model weights from checkpoint")
                except:
                    print("Could not load best weights, using current model")
                    best_model = hybrid_model
                break

        gc.collect()

    # If training completed without early stopping, make sure we use the best model
    if best_model is None or patience_counter >= patience:
        try:
            hybrid_model.load_weights(checkpoint_path)
            best_model = hybrid_model
            print("Loaded best model weights after training")
        except:
            print("Could not load best weights, using final model")
            best_model = hybrid_model

    # Plot training history
    plt.figure(figsize=(10, 6))
    plt.plot(epoch_losses, label='Training Loss', marker='x')
    plt.plot(epoch_val_losses, label='Validation Loss', marker='o')
    plt.title("Training vs Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('training_loss.png')
    plt.show()

    # Plot accuracy
    plt.figure(figsize=(10, 6))
    plt.plot(epoch_val_accs, label='Validation Accuracy', marker='o')
    plt.title("Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('validation_accuracy.png')
    plt.show()

    # CHANGED: Fixed evaluation on test dataset
    print("\nEvaluating on test dataset...")
    test_acc_metric = tf.keras.metrics.BinaryAccuracy()
    test_loss_total = 0
    test_batches = 0
    all_predictions = []
    all_labels = []

    for test_batch_idx, (x_batch, y_batch) in enumerate(test_dataset):
        with tf.device('/GPU:0'):
            test_logits = best_model(x_batch, training=False)
            test_loss = loss_fn(y_batch, test_logits)

        # Update the metric (don't reset after each batch)
        test_acc_metric.update_state(y_batch, test_logits)

        # Store predictions and labels for ROC curve
        all_predictions.extend(test_logits.numpy().flatten())
        all_labels.extend(y_batch.numpy().flatten())

        test_loss_total += test_loss.numpy()
        test_batches += 1

        if (test_batch_idx + 1) % test_chunk_size == 0:
            # CHANGED: Show progress with cumulative metrics
            current_test_loss = test_loss_total / test_batches
            current_test_acc = test_acc_metric.result().numpy()
            print(f"  Test progress - chunk {test_batch_idx // test_chunk_size + 1} cumulative loss: {current_test_loss:.4f}, accuracy: {current_test_acc:.4f}")
            gc.collect()

    # Final test metrics calculation after all batches
    test_loss = test_loss_total / max(1, test_batches)
    test_acc = test_acc_metric.result().numpy()

    print(f"Final test metrics - loss: {test_loss:.4f}, accuracy: {test_acc:.4f}")

    # Plot ROC curve
    from sklearn.metrics import roc_curve, auc
    fpr, tpr, _ = roc_curve(all_labels, all_predictions)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('roc_curve.png')
    plt.show()

    # Plot confusion matrix
    from sklearn.metrics import confusion_matrix
    import seaborn as sns

    # Convert predictions to binary
    binary_predictions = [1 if p >= 0.5 else 0 for p in all_predictions]
    cm = confusion_matrix(all_labels, binary_predictions)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    plt.show()

    return best_model


# Debug shapes and run
for x_batch, y_batch in train_dataset_tf.take(1):
    print("x_batch shape:", x_batch.shape)
    print("y_batch shape:", y_batch.shape)
    break

tf.debugging.set_log_device_placement(True)

# Run training with test dataset evaluation
best_model = train_model_with_chunks(train_dataset_tf, val_dataset_tf, test_dataset_tf, chunk_size=64, val_chunk_size=64, test_chunk_size=64)

