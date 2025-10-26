# Recurrent Neural Network (RNN) from Scratch

This project implements a **Recurrent Neural Network (RNN)** from scratch using PyTorch — without relying on `nn.RNN`.  
It demonstrates how recurrent architectures process sequential data step by step, forming the conceptual foundation for modern architectures like **Transformers**, which replace recurrence with **self-attention** for parallel sequence modeling.

---

## 🧠 Overview

- Built a **character-level RNN** for name language classification.  
- Implemented **manual forward propagation** and **Backpropagation Through Time (BPTT)**.  
- Applied **gradient clipping** for stable training.  
- Evaluated performance using **NLLLoss** and visualized training loss.  
- Compared baseline RNN behavior with modern recurrent variants like **GRU** and **LSTM** — precursors to the **Transformer** model.


## 🧩 Model Architecture

### Simple RNN Cell
```h_t = tanh(W_hh * h_{t-1} + W_xh * x_t)
y_t = log_softmax(W_hy * [x_t, h_t])
``` 
### Components
- **Input:** One-hot encoded characters  
- **Hidden layer:** `nn.Linear(input_size + hidden_size, hidden_size)`  
- **Output layer:** `nn.Linear(input_size + hidden_size, output_size)`  
- **Activation:** `tanh`  
- **Loss:** `NLLLoss`

---

## 🚀 Training

### Steps
1. Load and preprocess the dataset (`unicode → ASCII`, category parsing).  
2. Convert each name into one-hot vectors.  
3. Train using **SGD + momentum**, with **gradient clipping** to avoid exploding gradients.  
4. Log and plot the average loss.

### Example
```python
for i in range(n_iters):
    category, line, cat_tensor, line_tensor = random_training_example(category_lines, all_categories)
    output, loss = train(line_tensor, cat_tensor)
```
📊 Results

The RNN learns to classify names by language based on letter sequences.

Example outputs:
```
Input: "Albert"   →  French ✅  
Input: "Gonzalez" →  Spanish ✅
```
🧩 Interactive Prediction

After training, run:
```
python predict.py

```
Example:
```
Input: Marco
Output: Italian
```
🧪 Concepts Demonstrated

Recurrent computation over sequences

Backpropagation Through Time (BPTT)

Gradient clipping

Character-level modeling

Comparison between RNN, GRU, LSTM, and the conceptual shift toward Transformers

🧰 Requirements
```
pip install torch matplotlib
```
📚 References

PyTorch RNN Classification Tutorial

Understanding LSTM and RNNs

Attention Is All You Need (Vaswani et al., 2017)
