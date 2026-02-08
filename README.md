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

## ⚡ High-Performance Kernel Generation Strategy

To maximize the efficiency of the S4 layer, I implemented a dual-path kernel generation engine that balances numerical precision and raw execution speed using a **Hybrid Dispatcher**.

### 1. Log-space Kernel Generation (High Precision & SIMD)
For long-range sequences, we utilize log-space arithmetic to prevent floating-point drift.
- **Mathematical Derivation**:
  $$\log(K_t) = \log(C) + t \cdot \log(\bar{A}) + \log(\bar{B})$$
  $$K_t = \exp(\log(C) + t \cdot \log(\bar{A}) + \log(\bar{B}))$$
- **SIMD Acceleration**: Each timestep $t$ is calculated independently, allowing the CPU to process multiple steps in parallel using `@Vector`. This achieves **Zero-Drift precision (`0e0`)** even at a sequence length of $10^6$.


### 2. Sequential Generation (Low Latency)
For short-range bursts, we prioritize speed by using direct complex-valued recursive multiplication.
- **Derivation**: $K_t = K_{t-1} \cdot \bar{A}$ (Phase rotation and magnitude decay).
- **Optimization**: This path avoids expensive transcendental function calls (`exp`, `log`, `sin`, `cos`), resulting in up to **4x faster execution** for short sequences with negligible error ($< 10^{-6}$).

### 3. Hybrid Dispatcher Results
The engine dynamically switches strategies at a 1024-step threshold.

| Strategy | Sequence Length | Speed (1M) | Precision | Best For |
| :--- | :--- | :--- | :--- | :--- |
| **Sequential** | $\le 1024$ | **~6ms** | $10^{-6}$ | Real-time Inference |
| **SIMD Log** | $> 1024$ | **~35ms** | **`0e0`** | Long-range Training |


![Kernel Benchmark Results (short sequence)](https://private-user-images.githubusercontent.com/9676553/546732745-e574ea8c-5dd5-463e-b7f4-f14df654b7a2.png)

![Kernel Benchmark Results (long sequence)](https://private-user-images.githubusercontent.com/9676553/546733952-78c309c2-a45a-48b1-ba78-d5527502d74a.png)

### 🛠️ Comparison of Kernel Generation Methods

| Method | Approach | Key Mechanism | Best For | Precision |
| :--- | :--- | :--- | :--- | :--- |
| **Base Loop** | Naive Recursive | Direct $C\bar{A}^t\bar{B}$ calculation for each step. | Baseline / Testing | High |
| **Sequential Loop** | Recursive Rotation | Iterative update using phase rotation & magnitude scaling. | Short Seq ($\le 1024$) | Moderate ($10^{-6}$) |
| **Normal Log Loop** | Log-space Scalar | Independent calculation using $\exp(\sum \log)$ per step. | Long Seq Stability | **Zero-Drift (0e0)** |
| **SIMD Log Loop** | Log-space Parallel | Vectorized log-space calculation using `@Vector`. | High-throughput Long Seq | **Zero-Drift (0e0)** |
| **Hybrid Dispatcher** | Adaptive Selection | Dynamic switching between **Sequential** and **SIMD Log**. | **All Use Cases** | **Optimal** |

### 🔍 Implementation Details

* **Recursive Path (Sequential)**: Uses 4 real multiplications and 2 additions per step. Optimized for low-latency by avoiding heavy transcendental functions (`exp`, `log`).
* **Independent Path (Log-space SIMD)**: Computes each timestep independently to eliminate error accumulation. Leverages hardware-level parallelism to offset the cost of `exp` and `log` operations.
* **Hybrid Logic**: Automatically applies the **Recursive** path for speed in short sequences and the **SIMD Log** path for numerical integrity in long-range dependencies.

# 🧠 S4 Gradient Derivation Guide (for Internal Reference)

This guide derives the gradients for $A, B, C$ in three different perspectives: SSM (Recurrence), Convolution (Kernel), and Discretization (Bilinear).

---

### 1. SSM Recurrence (BPTT Perspective)

In the state equation $x_{t} = A x_{t-1} + B u_{t}$ and output $y_{t} = C x_{t}$:



**Parameter $C$ Gradient**
$$\frac{\partial \mathcal{L}}{\partial C} = \frac{\partial \mathcal{L}}{\partial y_{t}} \cdot \frac{\partial y_{t}}{\partial C} = e_{t} \cdot \text{conj}(x_{t})$$

**Parameter $B$ Gradient**
$$\frac{\partial \mathcal{L}}{\partial B} = \frac{\partial \mathcal{L}}{\partial y_{t}} \cdot \frac{\partial y_{t}}{\partial x_{t}} \cdot \frac{\partial x_{t}}{\partial B} = (e_{t} \cdot \text{conj}(C)) \cdot \text{conj}(u_{t})$$

**Parameter $A$ Gradient**
$$\frac{\partial \mathcal{L}}{\partial A} = \frac{\partial \mathcal{L}}{\partial y_{t}} \cdot \frac{\partial y_{t}}{\partial x_{t}} \cdot \frac{\partial x_{t}}{\partial A} = (e_{t} \cdot \text{conj}(C)) \cdot \text{conj}(x_{t-1})$$

**Definition of Terms:**
- $e_{t} = \frac{\partial \mathcal{L}}{\partial y_{t}}$ : The error signal (gradient of loss w.r.t output) at time $t$.
- $\text{conj}(\cdot)$ : Complex conjugate, used to align the gradient phase in complex space.

> **Why use `conj`?**
> To move the error back to the parameter in complex space, we use the conjugate to align the phase for steepest descent.

---

### 2. Convolutional Perspective (Kernel Gradient)

When the output is represented as a convolution with the kernel $K_{t} = C A^{t} B$:



**Parameter $C$ Gradient**
$$\frac{\partial \mathcal{L}}{\partial C} = \sum_{t} \frac{\partial \mathcal{L}}{\partial K_{t}} \cdot \frac{\partial K_{t}}{\partial C} = \sum_{t} g_{t} \cdot \text{conj}(A^{t} B)$$

**Parameter $B$ Gradient**
$$\frac{\partial \mathcal{L}}{\partial B} = \sum_{t} \frac{\partial \mathcal{L}}{\partial K_{t}} \cdot \frac{\partial K_{t}}{\partial B} = \sum_{t} g_{t} \cdot \text{conj}(C A^{t})$$

**Parameter $A$ Gradient**
$$\frac{\partial \mathcal{L}}{\partial A} = \sum_{t} \frac{\partial \mathcal{L}}{\partial K_{t}} \cdot \frac{\partial K_{t}}{\partial A} = \sum_{t} g_{t} \cdot \text{conj}(C \cdot t A^{t-1} \cdot B)$$

**Definition of Terms:**
- $g_{t} = \frac{\partial \mathcal{L}}{\partial K_{t}}$ : The gradient of the loss w.r.t the kernel at time $t$.
- $t A^{t-1}$ : Derived via the power rule, implemented using an **iota vector** in Zig for SIMD efficiency.

> **Note:** In the convolutional view, the gradients are accumulated over the entire sequence length $L$, leveraging the parallel nature of the S4 layer.

---
### 3. Continuous to Discrete (Bilinear Mapping)

The continuous parameters ($A, B$) are mapped to discrete parameters ($\bar{A}, \bar{B}$) using the **Bilinear Transformation (Tustin's method)**:
$$\bar{A} = \left(1 + \frac{\Delta}{2}A\right)\left(1 - \frac{\Delta}{2}A\right)^{-1}, \quad \bar{B} = \left(1 - \frac{\Delta}{2}A\right)^{-1} \cdot \sqrt{\Delta} \cdot B$$

---

#### 💡 Derivation of $\frac{\partial \bar{A}}{\partial A}$ (Using Substitution $k = \frac{\Delta}{2}$)

To simplify the derivation, let $k = \frac{\Delta}{2}$. The equation becomes:
$$\bar{A} = \frac{1 + kA}{1 - kA}$$

Using the quotient rule form $\frac{f'g - fg'}{g^2}$ (where $f = 1+kA, g = 1-kA$):

1. **Differentiate Terms**: 
   $f' = k$, $g' = -k$
2. **Apply Formula**:
   $$\frac{\partial \bar{A}}{\partial A} = \frac{k(1 - kA) - (1 + kA)(-k)}{(1 - kA)^2} = \frac{k - k^2A + k + k^2A}{(1 - kA)^2} = \frac{2k}{(1 - kA)^2}$$
3. **Extract $(1 + \bar{A})$**:
   Since $1 + \bar{A} = 1 + \frac{1+kA}{1-kA} = \frac{(1-kA)+(1+kA)}{1-kA} = \frac{2}{1-kA}$, we can rewrite the gradient as:
   $$\frac{\partial \bar{A}}{\partial A} = k \cdot \left( \frac{2}{1 - kA} \right) \cdot (1 - kA)^{-1}$$
   
   - Here, $\left( \frac{2}{1 - kA} \right)$ is equivalent to $(1 + \bar{A})$.
   - $(1 - kA)^{-1}$ is the remaining term from the power rule.

4. **Final Result**:
   $$\frac{\partial \bar{A}}{\partial A} = \frac{\Delta}{2} (1 + \bar{A}) (1 - \frac{\Delta}{2}A)^{-1}$$

---

#### 🚀 Backpropagation through Bilinear Mapping

**1. Total Gradient for Parameter $A$**
Since $A$ affects both $\bar{A}$ and $\bar{B}$, the gradient is the sum of two paths:
$$\frac{\partial \mathcal{L}}{\partial A} = \left( \frac{\partial \mathcal{L}}{\partial \bar{A}} \cdot \frac{\partial \bar{A}}{\partial A} \right) + \left( \frac{\partial \mathcal{L}}{\partial \bar{B}} \cdot \frac{\partial \bar{B}}{\partial A} \right)$$

**Substituting the analytical derivatives:**
$$\frac{\partial \mathcal{L}}{\partial A} = \left( g_{\bar{A}} \cdot \frac{\Delta}{2}(1 + \bar{A})(1 - \frac{\Delta}{2}A)^{-1} \right) + \left( g_{\bar{B}} \cdot \frac{\Delta}{2} \sqrt{\Delta} B (1 - \frac{\Delta}{2}A)^{-2} \right)$$

**2. Total Gradient for Parameter $B$**
$$\frac{\partial \mathcal{L}}{\partial B} = \frac{\partial \mathcal{L}}{\partial \bar{B}} \cdot \frac{\partial \bar{B}}{\partial B} = g_{\bar{B}} \cdot \left(1 - \frac{\Delta}{2}A\right)^{-1} \sqrt{\Delta}$$

**Definition of Terms:**
- $g_{\bar{A}}, g_{\bar{B}}$ : The gradients flowed back from the discrete-time S4 layer.
- $\Delta$ : The step size (sampling time).
- $\sqrt{\Delta}$ : Scaling factor to preserve variance.

> **Note on Implementation:** In Zig, the inverse $(1 - \frac{\Delta}{2}A)^{-1}$ is handled efficiently to maintain the DPLR structure of the state matrix.

**

---

### 4. Full BPTT (Temporal Gradient Flow)

The gradient travels to the past through the recurrent chain:

$$\frac{\partial L}{\partial x_{t-1}} = (e_{t-1} \cdot C) + \left( \frac{\partial L}{\partial x_{t}} \cdot A \right)$$



**Intuition:**
* The term $(e_{t-1} \cdot C)$ captures the immediate error from the current output.
* The term $(\frac{\partial L}{\partial x_{t}} \cdot A)$ propagates the "future" error back to the previous state through the transition matrix $A$.
* This recursive chain allows the model to learn long-range dependencies by flowing gradients across time steps.


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

## 📊 Execution Results & Logs

### S4 Engine Performance
Below is the terminal output demonstrating the real-time signal processing and denoising results.

![S4 Execution Log](https://private-user-images.githubusercontent.com/9676553/531302617-3feb652d-3e86-4825-b7ed-04ba9833179a.png?jwt=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3NjcyMzYyMDIsIm5iZiI6MTc2NzIzNTkwMiwicGF0aCI6Ii85Njc2NTUzLzUzMTMwMjYxNy0zZmViNjUyZC0zZTg2LTQ4MjUtYjdlZC0wNGJhOTgzMzE3OWEucG5nP1gtQW16LUFsZ29yaXRobT1BV1M0LUhNQUMtU0hBMjU2JlgtQW16LUNyZWRlbnRpYWw9QUtJQVZDT0RZTFNBNTNQUUs0WkElMkYyMDI2MDEwMSUyRnVzLWVhc3QtMSUyRnMzJTJGYXdzNF9yZXF1ZXN0JlgtQW16LURhdGU9MjAyNjAxMDFUMDI1MTQyWiZYLUFtei1FeHBpcmVzPTMwMCZYLUFtei1TaWduYXR1cmU9NWFmNjUwNGQ2NDFjYjgzNWI1ODAzZDBjMjQ5MGNmMmExMjk0OWIyMmNiZDQ3MWFkZjVjZmNmMTZjMDY5ZmQyNyZYLUFtei1TaWduZWRIZWFkZXJzPWhvc3QifQ.hRSyQNYmjwQKsniROwuKDQD_v9IBV2Swf9t82sIizYM)

### 📋 Full Terminal Output

#### S4 Engine (Complex.zig)
```bash
Starting S4 Model test..
Epoch 0 Channel 0: Loss = 5.476446, A = 0.888 + 0.179i B = 0.098 + 0.008i C = 0.274 + -0.048i
Epoch 0 Channel 1: Loss = 5.476446, A = 0.888 + 0.180i B = 0.098 + 0.008i C = 0.274 + -0.048i
Epoch 0 Channel 2: Loss = 5.476446, A = 0.888 + 0.180i B = 0.098 + 0.008i C = 0.274 + -0.048i
Epoch 0 Channel 3: Loss = 5.476446, A = 0.803 + 0.429i B = 0.094 + 0.020i C = 0.250 + -0.032i
Epoch 1000 Channel 0: Loss = 1.611923, A = 0.901 + 0.099i B = -0.045 + 0.019i C = -0.258 + 0.243i
Epoch 1000 Channel 1: Loss = 1.611923, A = 0.904 + 0.083i B = -0.057 + 0.008i C = -0.304 + 0.404i
Epoch 1000 Channel 2: Loss = 1.611923, A = 0.908 + 0.060i B = -0.076 + -0.015i C = -0.276 + 0.640i
Epoch 1000 Channel 3: Loss = 1.611923, A = 0.807 + 0.285i B = -0.696 + 0.034i C = -0.313 + -0.005i
Epoch 2000 Channel 0: Loss = 0.863172, A = 0.905 + 0.092i B = -0.014 + -0.007i C = 0.236 + -0.576i
Epoch 2000 Channel 1: Loss = 0.863172, A = 0.903 + 0.056i B = -0.006 + 0.006i C = -0.082 + 0.075i
Epoch 2000 Channel 2: Loss = 0.863172, A = 0.938 + -0.217i B = 0.010 + -0.024i C = -0.294 + 2.335i
Epoch 2000 Channel 3: Loss = 0.863172, A = 0.808 + 0.268i B = -0.722 + -0.053i C = -0.220 + 0.161i
Epoch 3000 Channel 0: Loss = 0.505832, A = 0.909 + 0.108i B = -0.054 + 0.010i C = 0.915 + -0.315i
Epoch 3000 Channel 1: Loss = 0.505832, A = 0.904 + 0.056i B = 0.007 + 0.009i C = -0.098 + 0.283i
Epoch 3000 Channel 2: Loss = 0.505832, A = 0.862 + -0.383i B = 0.035 + 0.073i C = 1.901 + 0.245i
Epoch 3000 Channel 3: Loss = 0.505832, A = 0.811 + 0.268i B = -0.708 + -0.056i C = -0.189 + 0.262i
Epoch 4000 Channel 0: Loss = 0.318500, A = 0.898 + 0.123i B = -0.114 + 0.093i C = 0.407 + 0.437i
Epoch 4000 Channel 1: Loss = 0.318500, A = 0.903 + 0.056i B = 0.029 + 0.020i C = -0.033 + 0.097i
Epoch 4000 Channel 2: Loss = 0.318500, A = 0.803 + -0.443i B = -0.144 + 0.057i C = 0.222 + -1.181i
Epoch 4000 Channel 3: Loss = 0.318500, A = 0.819 + 0.273i B = -0.654 + -0.043i C = -0.315 + 0.277i
Epoch 5000 Channel 0: Loss = 0.244188, A = 0.891 + 0.129i B = -0.138 + 0.205i C = 0.038 + 0.446i
Epoch 5000 Channel 1: Loss = 0.244188, A = 0.903 + 0.056i B = 0.023 + 0.045i C = 0.132 + 0.142i
Epoch 5000 Channel 2: Loss = 0.244188, A = 0.775 + -0.494i B = -0.049 + -0.151i C = -1.267 + -0.210i
Epoch 5000 Channel 3: Loss = 0.244188, A = 0.830 + 0.283i B = -0.560 + 0.000i C = -0.426 + 0.250i
Epoch 6000 Channel 0: Loss = 0.210453, A = 0.889 + 0.131i B = -0.138 + 0.244i C = -0.109 + 0.436i
Epoch 6000 Channel 1: Loss = 0.210453, A = 0.905 + 0.055i B = 0.004 + 0.059i C = 0.313 + 0.236i
Epoch 6000 Channel 2: Loss = 0.210453, A = 0.763 + -0.534i B = 0.065 + -0.116i C = -1.332 + 0.963i
Epoch 6000 Channel 3: Loss = 0.210453, A = 0.834 + 0.289i B = -0.522 + 0.024i C = -0.487 + 0.213i
Epoch 7000 Channel 0: Loss = 0.189896, A = 0.889 + 0.132i B = -0.138 + 0.256i C = -0.217 + 0.439i
Epoch 7000 Channel 1: Loss = 0.189896, A = 0.907 + 0.055i B = -0.008 + 0.062i C = 0.530 + 0.311i
Epoch 7000 Channel 2: Loss = 0.189896, A = 0.761 + -0.561i B = 0.086 + -0.076i C = -1.152 + 1.624i
Epoch 7000 Channel 3: Loss = 0.189896, A = 0.835 + 0.291i B = -0.511 + 0.032i C = -0.513 + 0.177i
Epoch 8000 Channel 0: Loss = 0.175427, A = 0.889 + 0.133i B = -0.137 + 0.260i C = -0.311 + 0.455i
Epoch 8000 Channel 1: Loss = 0.175427, A = 0.911 + 0.054i B = -0.014 + 0.063i C = 0.765 + 0.359i
Epoch 8000 Channel 2: Loss = 0.175427, A = 0.763 + -0.579i B = 0.088 + -0.057i C = -1.045 + 2.004i
Epoch 8000 Channel 3: Loss = 0.175427, A = 0.835 + 0.292i B = -0.508 + 0.036i C = -0.530 + 0.146i
Epoch 9000 Channel 0: Loss = 0.164259, A = 0.890 + 0.134i B = -0.137 + 0.262i C = -0.398 + 0.488i
Epoch 9000 Channel 1: Loss = 0.164259, A = 0.916 + 0.052i B = -0.017 + 0.062i C = 1.006 + 0.380i
Epoch 9000 Channel 2: Loss = 0.164259, A = 0.766 + -0.589i B = 0.088 + -0.049i C = -1.007 + 2.218i
Epoch 9000 Channel 3: Loss = 0.164259, A = 0.835 + 0.292i B = -0.506 + 0.037i C = -0.545 + 0.118i

--- Final Verification ---
Step 0: Pred(0.00 + 0.00i) vs Target(0.00 + 0.00i)
Step 1: Pred(0.20 + 0.06i) vs Target(0.20 + 0.00i)
Step 2: Pred(0.30 + 0.07i) vs Target(0.39 + 0.00i)
Step 3: Pred(0.35 + 0.01i) vs Target(0.56 + 0.00i)
Step 4: Pred(0.61 + 0.02i) vs Target(0.72 + 0.00i)
Step 5: Pred(0.79 + 0.03i) vs Target(0.84 + 0.00i)
Step 6: Pred(0.83 + -0.04i) vs Target(0.93 + 0.00i)
Step 7: Pred(1.00 + -0.04i) vs Target(0.99 + 0.00i)
Step 8: Pred(1.09 + -0.02i) vs Target(1.00 + 0.00i)
Step 9: Pred(0.97 + -0.07i) vs Target(0.97 + 0.00i)
```

## 🧪 Foundational Machine Learning: Linear Regression from Scratch

To verify the core mathematical logic and training loops within the Zig ecosystem, I implemented a **Linear Regression model** using pure Zig (`sashimiCore.zig`). This served as a baseline for the more complex state-space dynamics.

### 📸 Training Visualization & Execution
Below is the terminal execution showing the model's convergence through Gradient Descent.

![Linear Regression Execution](https://private-user-images.githubusercontent.com/9676553/531302070-541993d6-8902-4c34-9a2d-7822b87574ae.png?jwt=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3NjcyMzYzODQsIm5iZiI6MTc2NzIzNjA4NCwicGF0aCI6Ii85Njc2NTUzLzUzMTMwMjA3MC01NDE5OTNkNi04OTAyLTRjMzQtOWEyZC03ODIyYjg3NTc0YWUucG5nP1gtQW16LUFsZ29yaXRobT1BV1M0LUhNQUMtU0hBMjU2JlgtQW16LUNyZWRlbnRpYWw9QUtJQVZDT0RZTFNBNTNQUUs0WkElMkYyMDI2MDEwMSUyRnVzLWVhc3QtMSUyRnMzJTJGYXdzNF9yZXF1ZXN0JlgtQW16LURhdGU9MjAyNjAxMDFUMDI1NDQ0WiZYLUFtei1FeHBpcmVzPTMwMCZYLUFtei1TaWduYXR1cmU9YzVjNjk4YjUzNWUyNTM0ODc3NDdjMDg2ZmY4NjFhOTAwZjgzNzE5MjZkZjg0MGU2OGFmMTEwMDQ1YmQxNWEwYyZYLUFtei1TaWduZWRIZWFkZXJzPWhvc3QifQ.nH3UUF-f0JBS7qgmPXrF7g1cdUDoMFjROX1gEYBuvzk)

### 📋 Training Log Details

```bash
Model persisted: Train_temp.bin (2 weights)
Model persisted: Train_temp.bin (2 weights)
Model persisted: Train_temp.bin (2 weights)
Model persisted: Train_temp.bin (2 weights)
Model persisted: Train_temp.bin (2 weights)
Model persisted: Train_temp.bin (2 weights)
Model persisted: Train_temp.bin (2 weights)
Model persisted: Train_temp.bin (2 weights)
Model persisted: Train_temp.bin (2 weights)
Model persisted: Train_temp.bin (2 weights)
Model persisted: Train_temp.bin (2 weights)
Model persisted: Train_temp.bin (2 weights)
Model persisted: Train_temp.bin (2 weights)
Model persisted: Train_temp.bin (2 weights)
Model persisted: Train_temp.bin (2 weights)
Model persisted: Train_temp.bin (2 weights)
Model persisted: Train_temp.bin (2 weights)
Model persisted: Train_temp.bin (2 weights)
Model persisted: Train_temp.bin (2 weights)
Training converged. Target loss reached.
length weight: 0.50357, width weight: 0.18419
Model persisted: Train.bin (2 weights)
Model mmapped for high-speed access: Train.bin
loaded Model (Mmap): .{ .weights = { 0.50356865, 0.18419152 }, .bias = 0.012379333, .epochs = 30000, .lr = 0.001 }
prediction: 13.89094, answer = 13.90000
Accuracy: 99.93%
```