#include <iostream>

// ==========================================================================================
// || ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Helper Functions ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ||
// ==========================================================================================
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

// ==========================================================================================
// || ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Matrix Multiplication Implementations ~~~~~~~~~||
// ==========================================================================================
template <int rows, int cols, int inners>
void matMulNaive(const float *A, const float *B, float *C)
{
/*
    Naive implementation of matrix multiplication C = A * B
    where A is (rows x inners), B is (inners x cols), and C is (rows x cols).

    This provides a baseline for performance benchmarking of optimized versions.
*/

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

template <int rows, int cols, int inners>
void matmulNaiveRegister(const float *A, const float *B, float *C)
{
/*
    Keep dot product in a register to reduce memory accesses and improve latency.
*/
    // Perform matrix multiplication C = A * B
    for (int row = 0; row < rows; row++)
    {
        for (int col = 0; col < cols; col++)
        {
            int sum = 0;
            for (int inner = 0; inner < inners; inner++)
            {
                sum += A[row * inners + inner] * B[inner * cols + col];
            }
            C[row * cols + col] = sum;
        }
    }
}

template <int rows, int cols, int inners>
void matmulLoopReordered(const float *A, const float *B, float *C)
{
    // Perform matrix multiplication C = A * B
    for (int row = 0; row < rows; row++)
    {
        for (int inner = 0; inner < inners; inner++)
        {
            // int sum = 0;
            for (int col = 0; col < cols; col++)
            {
                C[row * cols + col] += A[row * inners + inner] * B[inner * cols + col];
            }
            // C[row * cols + col] = sum;
        }
    }
    return;
}


template <int rows, int cols, int inners, int tile_size>
void matMulTiling(const float *A, const float *B, float *C)
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

template <int rows, int cols, int inners, int tile_size>
void matMulRowColParrallelInnerTiling(const float *A, const float *B, float *C)
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

int main()
{
    // Matrix dimensions
    const int rows = 2;
    const int cols = 3;
    const int inners = 4;

    float A[rows * inners];
    float B[inners * cols];
    float C[rows * cols];

    // Initialize A and B with some values
    for (int i = 0; i < rows * inners; ++i) A[i] = i + 1;
    for (int i = 0; i < inners * cols; ++i) B[i] = i + 1;

    // Print matrix A
    printMatrix<rows, inners>(A, "Matrix A");

    // Print matrix B
    printMatrix<inners, cols>(B, "Matrix B");

    // Perform naive matrix multiplication and print results.
    initializeMatrix<rows, cols>(C);
    matMulNaive<rows, cols, inners>(A, B, C);
    std::cout << "Method: Naive Matrix Multiplication\n";
    printMatrix<rows, cols>(C, "Result Matrix C");

    // Perform naive matrix multiplication with register optimization and print results.
    initializeMatrix<rows, cols>(C);
    matmulNaiveRegister<rows, cols, inners>(A, B, C);
    std::cout << "Method: Naive Matrix Multiplication with Register Optimization\n";
    printMatrix<rows, cols>(C, "Result Matrix C");

    // Perform loop reordered matrix multiplication and print results.
    initializeMatrix<rows, cols>(C);
    matmulLoopReordered<rows, cols, inners>(A, B, C);
    std::cout << "Method: Loop Reordered Matrix Multiplication\n";
    printMatrix<rows, cols>(C, "Result Matrix C");


    // Perform tiled matrix multiplication and print results.
    const int tile_size = 2;
    initializeMatrix<rows, cols>(C);
    matMulTiling<rows, cols, inners, tile_size>(A, B, C);
    std::cout << "Method: Tiled Matrix Multiplication with tile size " << tile_size << "\n";
    printMatrix<rows, cols>(C, "Result Matrix C");

    // Perform parallel matrix multiplication and print results.
    initializeMatrix<rows, cols>(C);
    matMulRowColParrallelInnerTiling<rows, cols, inners, tile_size>(A, B, C);
    std::cout << "Method: Parallel Matrix Multiplication\n";
    printMatrix<rows, cols>(C, "Result Matrix C");

    return 0;
}