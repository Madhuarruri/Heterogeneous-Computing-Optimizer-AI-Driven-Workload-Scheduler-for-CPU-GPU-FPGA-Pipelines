
// OpenCL kernel for General Matrix Multiplication (GEMM)
// Optimized for AMD GPUs with local memory tiling

#define TS 16  // Tile size
#define WPT 4  // Work per thread

__kernel void gemm_nn(__global float* A,
                      __global float* B, 
                      __global float* C,
                      const int M,
                      const int N,
                      const int K,
                      const float alpha,
                      const float beta) {

    // Thread identifiers
    const int row = get_local_id(0); // Local row ID (max TS)
    const int col = get_local_id(1); // Local col ID (max TS)
    const int globalRow = TS*get_group_id(0) + row; // Row ID of C (0..M)
    const int globalCol = TS*get_group_id(1) + col; // Col ID of C (0..N)

    // Local memory to fit a tile of TS*TS elements of A and B
    __local float Asub[TS][TS];
    __local float Bsub[TS][TS];

    // Initialize the accumulation register
    float acc = 0.0f;

    // Loop over all tiles
    const int numTiles = K/TS;
    for (int t=0; t<numTiles; t++) {

        // Load one tile of A and B into local memory
        const int tiledRow = TS*t + row;
        const int tiledCol = TS*t + col;
        Asub[col][row] = A[tiledCol*M + globalRow];
        Bsub[col][row] = B[globalCol*K + tiledRow];

        // Synchronize to make sure the tile is loaded
        barrier(CLK_LOCAL_MEM_FENCE);

        // Perform the computation for a single tile
        for (int k=0; k<TS; k++) {
            acc += Asub[k][row] * Bsub[col][k];
        }

        // Synchronize before loading the next tile
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Store the final result in C
    C[globalCol*M + globalRow] = alpha * acc + beta * C[globalCol*M + globalRow];
}

// Vectorized GEMM kernel for better performance
__kernel void gemm_vectorized(__global float4* A,
                             __global float4* B,
                             __global float4* C,
                             const int M,
                             const int N, 
                             const int K) {

    const int globalRow = get_global_id(0);
    const int globalCol = get_global_id(1);

    if (globalRow >= M/4 || globalCol >= N/4) return;

    float4 sum = (float4)(0.0f, 0.0f, 0.0f, 0.0f);

    for (int k = 0; k < K/4; k++) {
        float4 a = A[globalRow * (K/4) + k];
        float4 b = B[k * (N/4) + globalCol];
        sum += a * b;
    }

    C[globalRow * (N/4) + globalCol] = sum;
}
