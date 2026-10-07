# Perceptron from Scratch

A hands-on implementation of a **Perceptron from scratch using Python, NumPy, Pandas, and Matplotlib**.

This project is part of my **Deep Learning Projects Lab** and focuses on understanding the fundamental working of a Perceptron, including weights, bias, activation, prediction, error calculation, learning rate, epochs, and parameter updates.

The project is developed in **Visual Studio Code** and executed using a Python virtual environment managed with **uv**.

---

## 📌 Overview

A **Perceptron** is one of the simplest forms of an artificial neural network and is primarily used for binary classification.

The Perceptron takes input features, calculates a weighted sum, adds a bias, applies an activation function, and generates a prediction.

```text
Input Features
      ↓
Weights
      ↓
Weighted Sum + Bias
      ↓
Activation Function
      ↓
Prediction
      ↓
Calculate Error
      ↓
Update Weights & Bias
      ↓
Repeat for Multiple Epochs
```

---

## 🎯 Project Objectives

The main goal of this project is to understand the fundamentals of neural networks by implementing the Perceptron learning process.

### Concepts covered

* Artificial neuron
* Perceptron
* Input features
* Weights
* Bias
* Weighted sum
* Activation function
* Prediction
* Error
* Learning rate
* Epoch
* Weight updates
* Bias updates
* Binary classification
* Linear decision boundary
* Linear separability

---

## 🧠 How the Perceptron Works

The Perceptron calculates a weighted sum of the input features.

$$
z = w_1x_1 + w_2x_2 + ... + w_nx_n + b
$$

Where:

* `x` = input features
* `w` = weights
* `b` = bias
* `z` = weighted sum

The weighted sum is then passed through an activation function to generate the final prediction.

For binary classification, the prediction can be represented as:

```text
if weighted_sum >= threshold:
    prediction = 1
else:
    prediction = 0
```

---

## 🔄 Training Process

The Perceptron learns by comparing its prediction with the actual target.

The training process follows these steps:

```text
1. Initialize weights and bias
2. Take input features
3. Calculate weighted sum
4. Add bias
5. Apply activation function
6. Generate prediction
7. Calculate error
8. Update weights
9. Update bias
10. Repeat for all training samples
11. Repeat for multiple epochs
```

The basic weight update can be represented as:

$$
w_{new} = w_{old} + \eta(y-\hat{y})x
$$

Where:

* `η` = learning rate
* `y` = actual value
* `ŷ` = predicted value
* `x` = input feature

---

## ⚙️ Key Concepts

### 1. Weights

Weights determine the contribution of each input feature to the prediction.

```text
Input × Weight
```

Different weights allow the model to assign different importance to different features.

---

### 2. Bias

Bias is an additional parameter added to the weighted sum.

```text
Weighted Sum = Σ(Input × Weight) + Bias
```

The bias helps shift the decision boundary.

---

### 3. Activation Function

The activation function converts the weighted sum into a prediction.

For a basic Perceptron:

```text
Weighted Sum
      ↓
Threshold
      ↓
0 or 1
```

---

### 4. Error

Error represents the difference between the actual output and the predicted output.

Example:

```text
Actual      = 1
Prediction  = 0

Error       = 1
```

The error is used to update the model's weights and bias.

---

### 5. Learning Rate

The learning rate controls the size of each parameter update.

```text
Low Learning Rate
        ↓
Small Updates
        ↓
Slower Learning

High Learning Rate
        ↓
Large Updates
        ↓
Potentially Unstable Learning
```

---

### 6. Epoch

An **epoch** represents one complete pass through the entire training dataset.

For example:

```text
Epoch 1 → Entire dataset
Epoch 2 → Entire dataset
Epoch 3 → Entire dataset
...
```

Training over multiple epochs allows the Perceptron to repeatedly adjust its parameters.

---

## 📐 Decision Boundary

A Perceptron learns a **linear decision boundary** between two classes.

For two-dimensional data, the decision boundary can be represented as a line.

```text
Class 1

 ●  ●  ●
   ●  ●

-------------------  Decision Boundary

     ○  ○
   ○  ○  ○

Class 0
```

For higher-dimensional data, this becomes a **hyperplane**.

---

## 🔬 Linear Separability

A single Perceptron can learn patterns when the classes are **linearly separable**.

For example:

```text
● ● ● ●

-------------------

○ ○ ○ ○
```

A straight line can separate the two classes.

However, a single Perceptron cannot solve non-linearly separable problems such as the classic **XOR problem**.

This limitation leads to the development of multilayer neural networks.

---

## 🛠️ Tech Stack

### Programming Language

* Python

### Libraries

* **NumPy** — numerical computations and array operations
* **Pandas** — data handling and manipulation
* **Matplotlib** — data visualization

### Development Environment

* Visual Studio Code
* Python Virtual Environment
* uv

### Implementation Approach

The Perceptron learning logic is implemented **from scratch** using Python and NumPy rather than using a pre-built Perceptron implementation from a machine-learning framework.

---

## 📁 Project Structure

```text
Perceptron/
│
├── .venv/
│
├── main.py
│
├── pyproject.toml
│
├── uv.lock
│
└── README.md
```

> The `.venv` directory should normally be excluded from Git using `.gitignore`.

---

# 🚀 Getting Started

## Prerequisites

Make sure the following are installed:

* Python
* VS Code
* uv

---

## 1. Clone the Repository

```bash
git clone https://github.com/Sanketdev77/Deep-Learning-Projects-Lab.git
```

Navigate to the Perceptron project:

```bash
cd Deep-Learning-Projects-Lab/Perceptron
```

---

## 2. Create a Virtual Environment

Create the virtual environment using `uv`:

```bash
uv venv
```

This creates a `.venv` directory.

---

## 3. Activate the Virtual Environment

### Windows

```bash
.venv\Scripts\activate
```

After activation, the terminal will indicate that the virtual environment is active.

---

## 4. Install Dependencies

Install the required libraries:

```bash
uv pip install numpy pandas matplotlib
```

---

## 5. Run the Project

Run the Python application using:

```bash
uv run python main.py
```

The Perceptron implementation will execute from `main.py`.

---

## 📊 Visualization

**Matplotlib** is used to visualize the data and/or Perceptron results.

Visualization helps understand:

* Data distribution
* Class separation
* Predictions
* Decision boundaries
* Model behavior

---

## 🧩 Project Workflow

```text
                 Dataset
                    │
                    ▼
             Data Preparation
                    │
                    ▼
        Initialize Weights & Bias
                    │
                    ▼
             Weighted Sum
                    │
                    ▼
          Activation Function
                    │
                    ▼
               Prediction
                    │
                    ▼
             Calculate Error
                    │
                    ▼
          Update Weights & Bias
                    │
                    ▼
             Next Sample
                    │
                    ▼
              Next Epoch
                    │
                    ▼
             Trained Model
```

---

## 💡 Why Build a Perceptron From Scratch?

Implementing a Perceptron from scratch helps build a strong understanding of what happens inside a neural network instead of treating the model as a black box.

The project demonstrates the fundamental learning cycle:

```text
Input
  ↓
Weights
  ↓
Computation
  ↓
Prediction
  ↓
Error
  ↓
Parameter Update
  ↓
Learning
```

These concepts form the foundation for understanding more advanced Deep Learning algorithms.

---

## ⚠️ Limitations

A single Perceptron has several limitations:

* Primarily designed for binary classification
* Produces a linear decision boundary
* Cannot learn non-linear relationships
* Cannot solve XOR using a single layer
* Not suitable for complex Deep Learning problems by itself

These limitations motivate the use of **multilayer neural networks**.

---

## 📚 Key Learning Outcomes

After completing this project, I gained an understanding of:

* How an artificial neuron works
* How a Perceptron performs classification
* How weights affect predictions
* The role of bias
* How activation functions work
* How prediction errors are calculated
* How the learning rate affects updates
* How epochs are used during training
* How weights and bias are updated
* What linear separability means
* How a decision boundary is formed
* Why a single Perceptron cannot solve XOR

---

## 🔮 Next Step

The natural progression from a Perceptron is:

```text
Perceptron
     ↓
Multilayer Perceptron
     ↓
Artificial Neural Network
     ↓
Forward Propagation
     ↓
Loss Functions
     ↓
Gradient Descent
     ↓
Backpropagation
     ↓
Deep Neural Networks
     ↓
CNN / RNN / LSTM / Transformers
```

---

## 👨‍💻 Author

**Sanket Zambare**

AI/ML & Generative AI Engineer

GitHub: [Sanketdev77](https://github.com/Sanketdev77)

---
