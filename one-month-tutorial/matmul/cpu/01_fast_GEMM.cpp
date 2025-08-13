#include <iostream>
#include <chrono>
#include <functional>
#include <string>
#include <vector>
#include <iomanip>
#include <fstream>

// ==========================================================================================
// || ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Helper Functions ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ||
// ==========================================================================================
struct AlgorithmTest
{
    std::function<void(const float*, const float*, float*)> matMulFunc;
    std::string methodName;
};

template <int rows, int cols>
void printMatrix(const float* matrix, const std::string& name) {
    std::string dimensions = "(" + std::to_string(rows) + "x" + std::to_string(cols) + ")";
    std::cout << name << " " << dimensions << ":\n";
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            std::cout << matrix[i * cols + j] << " ";
        }
        std::cout << "\n";
    }
    std::cout << "\n\n";
}

template <int rows, int cols>
void initializeMatrix(float *M)
{
    for (int i = 0; i < rows * cols; ++i) {
        M[i] = static_cast<float>(0.0); // Fill with some values
    }
}

template <int rows, int cols, int inners>
float testMatMulAlgorithm(
    const float* A,
    const float* B,
    float* C,
    std::function<void(const float*, const float*, float*)> matMulFunc,
    const std::string& methodName
)
{

    initializeMatrix<rows, cols>(C);

    auto start = std::chrono::high_resolution_clock::now();
    matMulFunc(A, B, C);
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<float, std::milli> duration = end - start;
    std::cout << "Method: " << methodName << "\n";
    std::cout << "Duration: " << duration.count() << " ms\n";

    printMatrix<rows, cols>(C, "Result Matrix C");

    return duration.count();
}

// // Add this single function after your existing template functions
// void runSizeTest(int size) {
//     std::cout << "\n=== Testing Size: " << size << "x" << size << " ===\n";
    
//     // Allocate dynamic arrays
//     float* A = new float[size * size];
//     float* B = new float[size * size];
//     float* C = new float[size * size];
    
//     // Initialize with simple values
//     for (int i = 0; i < size * size; ++i) {
//         A[i] = static_cast<float>(i + 1);
//         B[i] = static_cast<float>(i + 1);
//     }
    
//     // Time just the naive algorithm for different sizes
//     auto start = std::chrono::high_resolution_clock::now();
    
//     // Inline the naive algorithm to avoid template complications
//     for (int i = 0; i < size * size; ++i) C[i] = 0; // Initialize C
    
//     for (int row = 0; row < size; row++) {
//         for (int col = 0; col < size; col++) {
//             for (int inner = 0; inner < size; inner++) {
//                 C[row * size + col] += A[row * size + inner] * B[inner * size + col];
//             }
//         }
//     }
    
//     auto end = std::chrono::high_resolution_clock::now();
//     std::chrono::duration<float, std::milli> duration = end - start;
    
//     std::cout << "Size " << size << ": " << duration.count() << " ms\n";
    
//     // Cleanup
//     delete[] A;
//     delete[] B;
//     delete[] C;
// }

// ==========================================================================================
// || ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Matrix Multiplication Implementations ~~~~~~~~~||
// ==========================================================================================
// template <int rows, int cols, int inners>
// void matMulNaive(const float *A, const float *B, float *C)
// {
// /*
//     Naive implementation of matrix multiplication C = A * B
//     where A is (rows x inners), B is (inners x cols), and C is (rows x cols).

//     This provides a baseline for performance benchmarking of optimized versions.
// */

//     // Initialize C to zero
//     for (int i = 0; i < rows * cols; ++i) C[i] = 0;

//     // Perform matrix multiplication C = A * B
//     for (int row = 0; row < rows; row++)
//     {
//         for (int col = 0; col < cols; col++)
//         {
//             for (int inner = 0; inner < inners; inner++)
//             {
//                 C[row * cols + col] += A[row * inners + inner] * B[inner * cols + col];
//             }
//         }
//     }
// }

// template <int rows, int cols, int inners>
// void matmulNaiveRegister(const float *A, const float *B, float *C)
// {
// /*
//     Keep dot product in a register to reduce memory accesses and improve latency.
// */
//     // Perform matrix multiplication C = A * B
//     for (int row = 0; row < rows; row++)
//     {
//         for (int col = 0; col < cols; col++)
//         {
//             int sum = 0;
//             for (int inner = 0; inner < inners; inner++)
//             {
//                 sum += A[row * inners + inner] * B[inner * cols + col];
//             }
//             C[row * cols + col] = sum;
//         }
//     }
// }

// template <int rows, int cols, int inners>
// void matmulLoopReordered(const float *A, const float *B, float *C)
// {
//     // Perform matrix multiplication C = A * B
//     for (int row = 0; row < rows; row++)
//     {
//         for (int inner = 0; inner < inners; inner++)
//         {
//             // int sum = 0;
//             for (int col = 0; col < cols; col++)
//             {
//                 C[row * cols + col] += A[row * inners + inner] * B[inner * cols + col];
//             }
//             // C[row * cols + col] = sum;
//         }
//     }
//     return;
// }


// template <int rows, int cols, int inners, int tile_size>
void matMulTiling(const float *A, const float *B, float *C, int rows, int cols, int inners, int tile_size = 16)
{
    for (int innerTile = 0; innerTile < inners; innerTile += tile_size)
    {
        for (int row = 0; row < rows; row++)
        {
            int innerTileEnd = std::min(innerTile + tile_size, inners);
            for (int inner = innerTile; inner < innerTileEnd; inner++)
            {
                for (int col = 0; col < cols; col++)
                {
                    C[row * cols + col] += A[row * inners + inner] * B[inner * cols + col];
                }
            }
        }
    }
    return;
}

// template <int rows, int cols, int inners, int tile_size>
void matMulRowColParrallelInnerTiling(const float *A, const float *B, float *C, int rows, int cols, int inners, int tile_size = 16)
{
    int SIMD_width = 256;

    // Implement SIMD-optimized matrix multiplication here
    for (int rowTile = 0; rowTile < rows; rowTile += SIMD_width) // Assuming SIMD width of 256
    {
        for (int colTile = 0; colTile < cols; colTile += SIMD_width) // Assuming SIMD width of 256
        {
            for (int innerTile = 0; innerTile < inners; innerTile += tile_size)
            {
                int rowTileEnd = std::min(rowTile + SIMD_width, rows);
                int colTileEnd = std::min(colTile + SIMD_width, cols);
                int innerTileEnd = std::min(innerTile + tile_size, inners);
                
                for (int row = rowTile; row < rowTileEnd; row++)
                {
                    for (int inner = innerTile; inner < innerTileEnd; inner++)
                    {
                        for (int col = colTile; col < colTileEnd; col++)
                        {
                            C[row * cols + col] += A[row * inners + inner] * B[inner * cols + col];
                        }
                    }
                }
            }
        }
    }
}

// Make algorithms work with runtime dimensions
void matMulNaive(const float *A, const float *B, float *C, int rows, int cols, int inners)
{
    // Initialize C to zero
    for (int i = 0; i < rows * cols; ++i) C[i] = 0;

    // Perform matrix multiplication C = A * B
    for (int row = 0; row < rows; row++)
    {
        for (int col = 0; col < cols; col++)
        {
            for (int inner = 0; inner < inners; inner++)
            {
                C[row * cols + col] += A[row * inners + inner] * B[inner * cols + col];
            }
        }
    }
}

void matmulNaiveRegister(const float *A, const float *B, float *C, int rows, int cols, int inners)
{
    for (int row = 0; row < rows; row++)
    {
        for (int col = 0; col < cols; col++)
        {
            float sum = 0;
            for (int inner = 0; inner < inners; inner++)
            {
                sum += A[row * inners + inner] * B[inner * cols + col];
            }
            C[row * cols + col] = sum;
        }
    }
}

void matmulLoopReordered(const float *A, const float *B, float *C, int rows, int cols, int inners)
{
    // Initialize C to zero
    for (int i = 0; i < rows * cols; ++i) C[i] = 0;
    
    for (int row = 0; row < rows; row++)
    {
        for (int inner = 0; inner < inners; inner++)
        {
            for (int col = 0; col < cols; col++)
            {
                C[row * cols + col] += A[row * inners + inner] * B[inner * cols + col];
            }
        }
    }
}

// Runtime version of testing function
float testMatMulAlgorithmRuntime(
    const float* A,
    const float* B,
    float* C,
    int rows, int cols, int inners,
    std::function<void(const float*, const float*, float*, int, int, int)> matMulFunc,
    const std::string& methodName,
    bool printResult = false
)
{
    // Initialize C to zero
    for (int i = 0; i < rows * cols; ++i) {
        C[i] = 0.0f;
    }

    auto start = std::chrono::high_resolution_clock::now();
    matMulFunc(A, B, C, rows, cols, inners);
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<float, std::milli> duration = end - start;
    
    if (printResult) {
        std::cout << "Method: " << methodName << "\n";
        std::cout << "Duration: " << duration.count() << " ms\n";
        std::cout << "Matrix size: " << rows << "x" << cols << "\n\n";
    }

    return duration.count();
}

struct RuntimeAlgorithmTest
{
    std::function<void(const float*, const float*, float*, int, int, int)> matMulFunc;
    std::string methodName;
};

// int main()
// {
//     // Define test sizes
//     std::vector<int> testSizes = {64, 128, 256, 512, 1024, 2048};
    
//     std::vector<RuntimeAlgorithmTest> algorithms = {
//         {matMulNaiveRuntime, "Naive"},
//         {matmulNaiveRegisterRuntime, "Register Optimized"},
//         {matmulLoopReorderedRuntime, "Loop Reordered"}
//     };

//     // Results table
//     std::cout << std::setw(12) << "Size";
//     for (const auto& algo : algorithms) {
//         std::cout << std::setw(20) << algo.methodName;
//     }
//     std::cout << "\n" << std::string(12 + 20 * algorithms.size(), '-') << "\n";

//     for (int size : testSizes) {
//         const int rows = size;
//         const int cols = size;
//         const int inners = size;

//         // Allocate matrices
//         std::vector<float> A(rows * inners);
//         std::vector<float> B(inners * cols);
//         std::vector<float> C(rows * cols);

//         // Initialize with random values
//         for (int i = 0; i < rows * inners; ++i) A[i] = static_cast<float>(rand()) / RAND_MAX;
//         for (int i = 0; i < inners * cols; ++i) B[i] = static_cast<float>(rand()) / RAND_MAX;

//         std::cout << std::setw(12) << size;

//         // Test each algorithm
//         for (const auto& algo : algorithms) {
//             float runtime = testMatMulAlgorithmRuntime(
//                 A.data(), B.data(), C.data(),
//                 rows, cols, inners,
//                 algo.matMulFunc,
//                 algo.methodName,
//                 false  // Don't print individual results
//             );
            
//             std::cout << std::setw(20) << std::fixed << std::setprecision(3) << runtime;
//         }
//         std::cout << "\n";
//     }

//     return 0;
// }

void runSizeTest(int size, std::ofstream& csvFile) {
    std::cout << "\n=== Testing Size: " << size << "x" << size << " ===\n";
    
    // Allocate dynamic arrays
    float* A = new float[size * size];
    float* B = new float[size * size];
    float* C = new float[size * size];
    
    // Initialize with simple values
    for (int i = 0; i < size * size; ++i) {
        A[i] = static_cast<float>(i + 1);
        B[i] = static_cast<float>(i + 1);
    }
    
    // Time each algorithm for the current size
    auto start_naive = std::chrono::high_resolution_clock::now();
    matMulNaive(A, B, C, size, size, size);
    auto end_naive = std::chrono::high_resolution_clock::now();
    float duration_naive = std::chrono::duration<float, std::milli>(end_naive - start_naive).count();
    std::cout << "Naive: " << duration_naive << " ms\n";

    auto start_register = std::chrono::high_resolution_clock::now();
    matmulNaiveRegister(A, B, C, size, size, size);
    auto end_register = std::chrono::high_resolution_clock::now();
    float duration_register = std::chrono::duration<float, std::milli>(end_register - start_register).count();
    std::cout << "Naive Register: " << duration_register << " ms\n";

    auto start_reordered = std::chrono::high_resolution_clock::now();
    matmulLoopReordered(A, B, C, size, size, size);
    auto end_reordered = std::chrono::high_resolution_clock::now();
    float duration_reordered = std::chrono::duration<float, std::milli>(end_reordered - start_reordered).count();
    std::cout << "Loop Reordered: " << duration_reordered << " ms\n";

    auto start_tiling = std::chrono::high_resolution_clock::now();
    matMulTiling(A, B, C, size, size, size);
    auto end_tiling = std::chrono::high_resolution_clock::now();
    float duration_tiling = std::chrono::duration<float, std::milli>(end_tiling - start_tiling).count();
    std::cout << "Tiled: " << duration_tiling << " ms\n";

    auto start_parallel = std::chrono::high_resolution_clock::now();
    matMulRowColParrallelInnerTiling(A, B, C, size, size, size);
    auto end_parallel = std::chrono::high_resolution_clock::now();
    float duration_parallel = std::chrono::duration<float, std::milli>(end_parallel - start_parallel).count();
    std::cout << "Parallel with Inner Tiling: " << duration_parallel << " ms\n";

    // Write to CSV file
    csvFile << size << "," << duration_naive << "," << duration_register << "," 
            << duration_reordered << "," << duration_tiling << "," << duration_parallel << "\n";

    // Cleanup
    delete[] A;
    delete[] B;
    delete[] C;
}

int main()
{
    // // Matrix dimensions
    // const int rows = 2;
    // const int cols = 3;
    // const int inners = 4;
    // const int tile_size = 2; // For tiled implementations

    // float A[rows * inners];
    // float B[inners * cols];
    // float C[rows * cols];

    // // Initialize A and B with some values
    // for (int i = 0; i < rows * inners; ++i) A[i] = i + 1;
    // for (int i = 0; i < inners * cols; ++i) B[i] = i + 1;

    // // Print matrix A
    // printMatrix<rows, inners>(A, "Matrix A");

    // // Print matrix B
    // printMatrix<inners, cols>(B, "Matrix B");

    // std::vector<AlgorithmTest> algorithms =
    // {
    //     {
    //         [](const float* A, const float* B, float* C) {
    //             matMulNaive<rows, cols, inners>(A, B, C);
    //         },
    //         "Naive Matrix Multiplication"
    //     },
    //     {
    //         [](const float* A, const float* B, float* C) {
    //             matmulNaiveRegister<rows, cols, inners>(A, B, C);
    //         },
    //         "Naive Matrix Multiplication with Register Optimization"
    //     },
    //     {
    //         [](const float* A, const float* B, float* C) {
    //             matmulLoopReordered<rows, cols, inners>(A, B, C);
    //         },
    //         "Loop Reordered Matrix Multiplication"
    //     },
    //     {
    //         [](const float* A, const float* B, float* C) {
    //             matMulTiling<rows, cols, inners, tile_size>(A, B, C);
    //         },
    //         "Tiled Matrix Multiplication"
    //     },
    //     {
    //         [](const float* A, const float* B, float* C) {
    //             matMulRowColParrallelInnerTiling<rows, cols, inners, tile_size>(A, B, C);
    //         },
    //         "Parallel Matrix Multiplication with Inner Tiling"
    //     }
    // };

    // std::vector<float> runtimes;

    // // test each algorithm
    // for (const auto& test : algorithms)
    // {
    //     float runtime = testMatMulAlgorithm<rows, cols, inners>(
    //         A, B, C, test.matMulFunc, test.methodName
    //     );
    //     runtimes.push_back(runtime);
    // }

    // // Optional: Print performance summary
    // std::cout << "\n=== Performance Summary ===\n";
    // for (size_t i = 0; i < algorithms.size(); ++i)
    // {
    //     std::cout << algorithms[i].methodName << ": " << runtimes[i] << " ms\n";
    // }

    std::cout << "\n" << std::string(50, '=') << "\n";
    std::cout << "SCALING ANALYSIS\n";
    std::cout << std::string(50, '=') << "\n";
    
    // Create CSV file
    std::ofstream csvFile("matrix_benchmark_results.csv");
    csvFile << "Size,Naive,NaiveRegister,LoopReordered,Tiled,ParallelTiling\n";

    std::vector<int> testSizes;
    for (int i = 1; i <= 12; ++i) { // 2^1=2, 2^12=4096
        testSizes.push_back(1 << i);
    }
    
    for (int size : testSizes) {
        runSizeTest(size, csvFile);
    }

    csvFile.close();
    std::cout << "Benchmark results saved to matrix_benchmark_results.csv\n";
    return 0;
}