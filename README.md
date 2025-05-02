# Q-ViX: Hybrid Quantum-Classical Deepfake Detection Model

Q-ViX is a hybrid deep learning model designed for high-accuracy image classification, specifically deepfake detection. It integrates classical architectures (Xception and Vision Transformer) with a custom Quantum Neural Network (QNN) built using Pennylane. This fusion enhances spatial and contextual feature extraction, offering strong generalization across diverse manipulated media datasets.

## Features

- Hybrid architecture combining:
  - **Xception** for spatial feature extraction
  - **Vision Transformer (ViT)** for global context modeling
  - **Quantum Neural Network (QNN)** via Pennylane for quantum-enhanced learning
- Distributed training with TensorFlow's `MirroredStrategy` for scalable performance
- Tested on high-resolution deepfake datasets (FaceForensics++, DFDC)
- Techniques transferable to medical imaging (X-ray, MRI, CT) and other high-stakes classification tasks


Make sure to download and preprocess these datasets before training.

## Requirements

- Python 3.8+
- TensorFlow 2.x
- Pennylane
- Transformers (Hugging Face)
- NumPy, Matplotlib, Scikit-learn



## Results

- Accuracy: **95%+** on test sets
- High generalization to unseen data
- Robust to common video compression artifacts

## Applications

While built for deepfake detection, Q-ViX is adaptable to other image classification tasks, especially in:
- Medical Imaging (X-ray, MRI, CT scan analysis)
- Security and Surveillance
- Digital Forensics

## License

This project is open-source under the MIT License.

---
