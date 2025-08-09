# Intro
This repository documents self learning when participating in a 30-day GPU development challenge. The challenge is dervied from mateiral hosted at [https://github.com/rkinas/daily-cuda](https://github.com/rkinas/daily-cuda). The exercises and solutions provided focus on comparing runtime analysis and model acuracy. A simple script will be developed to automate this process and each exercise will track the solution used to illustrate it's evolution. as needed. Each exercise will include a README.md file to provided further context of the instructions and solution.

---

## Challenges
This section outlines the daily exercises for the GPU challenge. Please see the read me of the correct file for further information. (WIP)

### Day 1-10
1. "Hello, GPU World" - Write and execute your first CUDA kernel.
2. Element-wise operations - Perform element-wise addition and scaling for tensors.
3. Activiation Functions - Implement ReLU, Sigmoid and Tanh activiations in CUDA.
4. Softmax Function - Compute the softmax for a 1D tensor.
5. Matrix-Vector Multiplication - Multiply a matrix and a vector efficiently.
6. Shared memory basics - Use shared memory to optimize tensor opertation.
7. Prefix Sum (scan) - Implement an inclusive prefix sum for a small tensor.
8. Loss Function: MSE - Computer the Mean Squared Error loss for two tensor.
9. Cross-entropy Loss - Implement the categorical cross-entropy loss function.
10. Memory Coalescing - Optimize memory acces for tensor addition.

### Day 11-20
11. Batch Matrix Multiplication - Multiply batches of matrices in parallel.
12. Tiled Matrix Multiplication - Optimize matrix multiplication usiing shared memory.
13. Forward Pass: Dense Layer - Implement a dense (fully connected) layer forward pass.
14. Backpropogation: Gradient of MSE - Compute the gradient of MSR loss with respect to weights.
15. Conolution: 1D Signal - Apply a 1D convolution to a signal
16. Convolution 2D Images - Implement a 2D convolution for image data.
17. Max Pooling - Apply max pooling to a 2D image.
18. Dropout - Simulate dropout regularization in a forward pass.
19. Layer Normalization - Implement layer normalization for a batch of inputs. 
20. Data Augmentation - Apply basic data augmentation (flipping cropping) on images.

### Day 21-30
21. Backpropagation: Dense Layer - Compute gradients for a dense layer.
22. Softmax with Logits - Implement a numerically stable softmax function with logits.
23. Sparse Matrix Operations - Multiply sparse matrices with dense tensors.
24. Custom CUDA Kernel for Attention - Implement a simplified attention mechancism.
25. Optimizier: SGD - Implement Stochastic Gradient Descent in CUDA.
26. Optimizer: Adam - Implement the Adam optimizer for model training. 
27. Embedding Lookup - Optimize embedding lookups for model training. 
28. Transformer Encoder - Implement the core computations of a transformer encoder block. 
29. Mixed Precision Training - Use half-precision (FP16) arithmetic for tensor computations.
30. Neural Network Interface - Optimize a simple neural network for real-time interface.

---
## Prerequisites
- Basic knowledge of C/C++ programming
- NVIDIA GPU with CUDA support
- Installed CUDA Toolkit (compatible version for your GPU)

---
## How to Use
1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/cuda-30-day-challenge.git
   ```
2. Navigate to the day-specific folder:
   ```bash
   cd day-01
   ```
3. Compile and run the code:
   ```bash
   nvcc main.cu -o main && ./main
   ```
---

## Resources
- [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html)
- [Deep Learning with CUDA](https://developer.nvidia.com/deep-learning)
- [CUDA Samples](https://github.com/NVIDIA/cuda-samples)
- [Accelerated Computing Documentation](https://developer.nvidia.com/documentation/)

---
## Contributing
Contributions are welcome! If you want to improve the solutions or suggest new challenges, feel free to open an issue or a pull request.
