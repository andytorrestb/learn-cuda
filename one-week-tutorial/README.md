# Additional Exercises to Material Found in 1-week Tutorial

Content is focused on developing runtime analysis for practice problems found in the source material of the 1-week tutorial. It is seemingly missing some chapter, but that is because the source material 
does not provide practive problems for those chapters.

# Chapter 4: Hello Cuda

Develop a nested for loop to evaluate runtime of kernel execution for Hello World. These results are trivial but they provide the logic used in later studies.

# Chapter 7: Vector Addition (CPU vs GPU)

Develop a for loop to evaluate runtime of elements wise vector addition using CPU and GPU Hardware.

# Chapter 9: Unified Memory

This tutorial attempts demo the built CUDA profiling CLI tools. This does not work on my current WSL configuration. I will re-test soon on an Ubuntu system. 

# Chapter 10: Pinned Memory

This tutorial explains the "pinned memory" concept to speed up CPU-GPU data transfer. Essentially removes a middle-man step when compared to pageable memory. 

# Chapter 11: Streaming

This tutorial uses a runtime study to explore synchronous and asynchronous data transfer. Asynchronous data transfer breaks down large operations into independent stream in order to mitigate dead time of individual processing cores. Thus improving latency.

