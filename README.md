# High-Performance S4 Model Implementation in Zig

This repository contains a from-scratch implementation of the **Structured State Space (S4)** model, optimized using **SIMD instructions** in the Zig programming language. The project demonstrates the effectiveness of S4 in sequence modeling and signal filtering through manual gradient derivation and optimization.

## 🚀 Key Features

- **S4Layer Architecture**: Implements the convolutional representation of State Space Models (SSMs), allowing for parallelizable sequence processing.
- **Manual BPTT & Complex Gradients**: Full implementation of **Truncated Backpropagation Through Time (BPTT)** with complex-valued gradient descent, avoiding reliance on deep learning frameworks.
- **SIMD Acceleration**: Leverages hardware-level parallelism for complex-valued convolutions and vector operations to ensure high-performance execution.
- **Continuous-to-Discrete Mapping**: Includes a robust discretization module using the **Bilinear Transformation (Tustin's method)** to map continuous system dynamics to discrete kernels.
- **Multi-channel Signal Filtering**: Successfully trained to filter high-frequency noise from sinusoidal signals, proving the model's ability to capture long-range dependencies and system stability.

## 🛠 Mathematical Implementation

- **State Transition**: $x_t = \bar{A}x_{t-1} + \bar{B}u_t$
- **Output Projection**: $y_t = Cx_t$
- **Kernel Generation**: $K = [CB, CAB, CA^2B, \dots, CA^{L-1}B]$
- **Optimization**: L2 Loss minimization using manual partial derivatives w.r.t. Complex parameters $A, B, C$.

## 📊 Results: Signal Denoising Success

The model was tested on a sequence of noisy sinusoidal signals. By updating the latent states and constraining the spectral radius of the transition matrix, the model successfully recovered the original signal with high fidelity.

**Final Verification Output:**
- Predicted values closely track the target phase and amplitude.
- System stability is maintained through weight normalization ($|A| < 1.0$).

## 🧪 Additional Implementations

### 1. Linear Regression from Scratch (`sashimiCore.zig`)
As a foundation for more complex state-space modeling, I implemented a **Linear Regression model** using pure Zig.
- **Core Logic:** Implemented Gradient Descent optimization without external math libraries.
- **Feature:** Demonstrates memory-efficient data handling and basic predictive modeling within the Zig ecosystem.

### 2. High-Performance DB API
To showcase full-stack integration and backend engineering skills, I developed a custom DB API.
- **Role:** Handles real-time data persistence and retrieval for signal processing results.
- **Focus:** Optimized for low-latency data flow between the processing engine and the storage layer.

## 💻 Technical Stack
- **Language**: Zig (0.16.0-dev.1484 or higher)
- **Compiler**: Zig Nightly Build (Leveraging latest language features)
- **Paradigm**: Low-level Systems Programming / Manual Memory Management
- **Optimization**: Native SIMD vectorization (@Vector)

## 🚀 How to Run

Ensure you have **Zig 0.16.0-dev** installed.

### Run S4 Engine (Main Research)
```bash
zig run Complex.zig
```

### Run Linear Regression Model
```bash
zig run sashimiCore.zig
```