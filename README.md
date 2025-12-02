# FT-Transformer for Network Intrusion Detection

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![License](https://img.shields.io/badge/license-MIT-green)

## Project Overview
This repository contains a TensorFlow/Keras implementation of the **FT-Transformer (Feature Tokenizer + Transformer)** architecture applied to the **UNSW-NB15** dataset for network intrusion detection.

The project demonstrates how to adapt "Transformer" architectures (originally designed for NLP) to tabular data by treating features as tokens. It solves the problem of detecting various network attacks (Exploits, Fuzzers, Generic) against normal traffic.

## Model Architecture
The model follows the standard FT-Transformer design suitable for tabular data.

### Key Components:

1.  **Feature Tokenizer:**
    * **Categorical Features:** Processed via `Embedding` layers.
    * **Numerical Features:** Projected via `Dense` layers with `LayerNormalization`.
    * *Result:* Transforms heterogeneous table columns into a uniform sequence of embeddings.

2.  **[CLS] Token Mechanism:**
    * A custom layer `AddCLSToken` appends a learnable vector to the beginning of the sequence.
    * Similar to BERT, this token aggregates information from all other features via the Self-Attention mechanism and is used for the final classification.

3.  **Transformer Encoder:**
    * Uses **Multi-Head Self-Attention** to capture complex interactions between features (e.g., how *Source IP* relates to *Packet Count*).
    * Includes Residual Connections (`Add`) and Feed-Forward Networks (FFN).

4.  **Preprocessing & Balancing:**
    * **SMOTE:** Applied to handle class imbalance in the training set.
    * **Scaling:** Numerical features are standardized using `StandardScaler`.

## Code Structure & Implementation

The core logic is implemented in `ft_transformer.py`. Below is a breakdown of the custom classes:

### `class AddCLSToken(layers.Layer)`
Implements the initialization and appending of the class token.
* **Role:** Acts as a global context aggregator.
* **Logic:** Creates a trainable weight `(1, 1, embedding_dim)`, tiles it to match the batch size, and concatenates it with the input feature tokens.

### `create_ft_transformer(...)`
Builds the model graph using the Keras Functional API.
* **Input:** Accepts separate inputs for each column.
* **Attention:** Uses `layers.MultiHeadAttention` with `key_dim=embedding_dim // num_heads`.
* **Regularization:** Applies `L2` regularization and `Dropout` to prevent overfitting.
* **Output:** Extracts the `[CLS]` token (index 0) and passes it through a Dense layer for Softmax classification.

## Performance
The model is trained over 3 independent runs to ensure stability.
* **Optimizer:** `AdamW` (with Weight Decay).
* **Loss Function:** Categorical Crossentropy.

![Training Metrics](./images/training_metrics.png)

## Installation & Usage

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/YOUR_USERNAME/FT-Transformer-NIDS.git](https://github.com/YOUR_USERNAME/FT-Transformer-NIDS.git)
    cd FT-Transformer-NIDS
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the training script:**
    ```bash
    python ft_transformer.py
    ```

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
