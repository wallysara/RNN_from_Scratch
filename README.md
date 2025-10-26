
This project implements a **Recurrent Neural Network (RNN)** from scratch using only PyTorch linear layers and tensor operations — without relying on high-level modules like `nn.RNN`. The model classifies names by language (e.g., English, French, Italian) using the **Names dataset** from the official PyTorch tutorial.

---

## Overview
- Manual **RNN cell** with hidden state propagation and **BPTT**
- **Gradient clipping** for stability
- Character-level inputs and **NLLLoss** objective
- Training loss tracking and **interactive prediction** mode

---

## Project Structure

---

## Model
- Input: one-hot vectors over allowed characters
- Hidden: `nn.Linear(input_size + hidden_size, hidden_size)` with `tanh`
- Output: `nn.Linear(input_size + hidden_size, n_categories)` + `LogSoftmax`
- Loss: `NLLLoss`

---

## Training
1) Load and preprocess data (Unicode → ASCII, category parsing)  
2) Convert names to tensors (`[seq_len x 1 x n_letters]`)  
3) Train with SGD + momentum and **clip_grad_norm_**  
4) Track loss every 1k steps, print status every 5k

Example:
```python
for i in range(n_iters):
    category, line, cat_tensor, line_tensor = random_training_example(category_lines, all_categories)
    output, loss = train(line_tensor, cat_tensor)
Inference
python predict.py

```
Example:
```
Input: Marco
Output: Italian
```
Requirements:
```
pip install torch matplotlib
```
References:
PyTorch Char-RNN Tutorial
Understanding LSTMs (colah blog)

Resume Snippet:
RNN from Scratch (Character-Level Classification) — Personal Project (2025)
• Implemented a pure-PyTorch RNN cell with BPTT and gradient clipping.
• Trained on character sequences for language classification of names; tracked loss and enabled interactive predictions.
