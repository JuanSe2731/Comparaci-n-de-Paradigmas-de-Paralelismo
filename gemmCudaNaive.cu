// gemmCudaNaive.cu — GEMM con CUDA (naive, sin shared memory)
// Compilar: nvcc -O2 -o matmul_cuda gemmCudaNaive.cu
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <algorithm>
#include <vector>
#include <cuda_runtime.h>

// ─── Configuración del protocolo ───
#define NREP 5
#define BLOCK_SIZE 16

// ─── Macro para verificar errores CUDA ───
#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "Error CUDA en %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(err)); \
        exit(1); \
    } \
} while(0)

// ─── Kernel CUDA naive ───
__global__ void gemm_cuda_naive(const float* A, const float* B, float* C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < N; k++) {
            sum += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

// ─── Kernel serial en host (para verificación) ───
void matmul_serial(const float* A, const float* B, float* C, int N) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            float acc = 0.0f;
            for (int k = 0; k < N; k++) {
                acc += A[i*N + k] * B[k*N + j];
            }
            C[i*N + j] = acc;
        }
    }
}

// ─── Inicialización determinista ───
void init_matrices(float* A, float* B, int N) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            A[i*N + j] = (float)(i + j) / N;
            B[i*N + j] = (float)(i - j + N) / N;
        }
    }
}

int main(int argc, char** argv) {
    // ─── Parseo de argumentos ───
    int N = 1024;
    if (argc > 1) N = atoi(argv[1]);

    printf("============================================\n");
    printf("  MULTIPLICACIÓN DE MATRICES — CUDA Naive\n");
    printf("============================================\n");
    printf("Tamaño de matriz: %d x %d\n", N, N);
    printf("Repeticiones: %d (se reporta la mediana)\n", NREP);

    // ─── Información del dispositivo ───
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("GPU: %s\n", prop.name);
    printf("Compute Capability: %d.%d\n", prop.major, prop.minor);
    printf("Multiprocesadores (SM): %d\n", prop.multiProcessorCount);
    printf("Memoria global: %.0f MB\n", prop.totalGlobalMem / (1024.0 * 1024.0));
    printf("--------------------------------------------\n");

    // ─── Asignación de memoria host ───
    size_t bytes = (size_t)N * N * sizeof(float);
    float* h_A = (float*)malloc(bytes);
    float* h_B = (float*)malloc(bytes);
    float* h_C = (float*)malloc(bytes);

    if (!h_A || !h_B || !h_C) {
        fprintf(stderr, "Error: No se pudo asignar memoria en host\n");
        return 1;
    }

    // ─── Inicialización determinista ───
    init_matrices(h_A, h_B, N);

    // ─── Asignación de memoria device ───
    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, bytes));
    CUDA_CHECK(cudaMalloc(&d_B, bytes));
    CUDA_CHECK(cudaMalloc(&d_C, bytes));

    // Copiar A y B al device
    CUDA_CHECK(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice));

    // ─── Configuración de lanzamiento ───
    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim((N + BLOCK_SIZE - 1) / BLOCK_SIZE,
                 (N + BLOCK_SIZE - 1) / BLOCK_SIZE);

    printf("Configuración de lanzamiento:\n");
    printf("  Grid:  %d x %d bloques\n", gridDim.x, gridDim.y);
    printf("  Bloque: %d x %d hilos\n", blockDim.x, blockDim.y);
    printf("  Hilos totales: %d\n", gridDim.x * gridDim.y * BLOCK_SIZE * BLOCK_SIZE);
    printf("--------------------------------------------\n");

    // ─── Calentamiento (1 ejecución sin medir) ───
    printf("Ejecutando calentamiento...\n");
    gemm_cuda_naive<<<gridDim, blockDim>>>(d_A, d_B, d_C, N);
    CUDA_CHECK(cudaDeviceSynchronize());

    // ─── Medición: NREP repeticiones con cudaEvent ───
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    std::vector<double> tiempos(NREP);
    printf("Ejecutando %d repeticiones...\n", NREP);

    for (int rep = 0; rep < NREP; rep++) {
        // Limpiar C en device
        CUDA_CHECK(cudaMemset(d_C, 0, bytes));

        CUDA_CHECK(cudaEventRecord(start));
        gemm_cuda_naive<<<gridDim, blockDim>>>(d_A, d_B, d_C, N);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        tiempos[rep] = (double)ms;
        printf("  Rep %d: %.4f ms\n", rep + 1, ms);
    }

    // Copiar resultado de vuelta al host
    CUDA_CHECK(cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost));

    // ─── Calcular mediana ───
    std::sort(tiempos.begin(), tiempos.end());
    double t_ms = tiempos[NREP / 2];

    // ─── Calcular GFLOPS ───
    double gflops = 2.0 * (double)N * N * N / (t_ms * 1e6);

    printf("--------------------------------------------\n");
    printf("Mediana: %.4f ms\n", t_ms);
    printf("Rendimiento: %.4f GFLOPS\n", gflops);

    // ─── Verificación vs serial (N <= 1024) ───
    double max_error = 0.0;
    if (N <= 1024) {
        printf("Verificando contra kernel serial...\n");
        float* C_ref = (float*)malloc(bytes);
        if (C_ref) {
            matmul_serial(h_A, h_B, C_ref, N);
            for (int i = 0; i < N*N; i++) {
                double err = fabs((double)h_C[i] - (double)C_ref[i]);
                if (err > max_error) max_error = err;
            }
            printf("Error máximo absoluto: %.6e\n", max_error);
            if (max_error < 1e-3)
                printf("✓ Verificación PASÓ (error < 1e-3)\n");
            else
                printf("✗ Verificación FALLÓ (error >= 1e-3)\n");
            free(C_ref);
        }
    } else {
        printf("Verificación omitida (N > 1024)\n");
    }

    // ─── Línea CSV ───
    printf("--------------------------------------------\n");
    printf("CSV: CUDA_NAIVE,%d,%.4f,%.4f,%.6e\n", N, t_ms, gflops, max_error);
    printf("============================================\n");

    // ─── Limpieza ───
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}