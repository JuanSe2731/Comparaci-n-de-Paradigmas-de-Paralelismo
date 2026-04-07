// matmul_omp.cpp — Multiplicación de matrices con OpenMP (CPU)
// Compilar: g++ -O2 -fopenmp -o matmul_omp matmul_omp.cpp
#include <omp.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <algorithm>
#include <vector>

// ─── Configuración del protocolo ───
#define NREP 5

// ─── Kernel serial (para verificación) ───
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

// ─── Kernel OpenMP CPU ───
void matmul_omp(const float* A, const float* B, float* C, int N) {
    #pragma omp parallel for collapse(2) schedule(static)
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
    printf("  MULTIPLICACIÓN DE MATRICES — OpenMP CPU\n");
    printf("============================================\n");
    printf("Tamaño de matriz: %d x %d\n", N, N);
    printf("Hilos OpenMP: %d\n", omp_get_max_threads());
    printf("Repeticiones: %d (se reporta la mediana)\n", NREP);
    printf("--------------------------------------------\n");

    // ─── Asignación de memoria ───
    size_t bytes = (size_t)N * N * sizeof(float);
    float* A = (float*)malloc(bytes);
    float* B = (float*)malloc(bytes);
    float* C = (float*)malloc(bytes);
    float* C_ref = NULL;

    if (!A || !B || !C) {
        fprintf(stderr, "Error: No se pudo asignar memoria\n");
        return 1;
    }

    // ─── Inicialización determinista ───
    init_matrices(A, B, N);

    // ─── Calentamiento (1 ejecución sin medir) ───
    printf("Ejecutando calentamiento...\n");
    matmul_omp(A, B, C, N);

    // ─── Medición: NREP repeticiones con omp_get_wtime() ───
    std::vector<double> tiempos(NREP);
    printf("Ejecutando %d repeticiones...\n", NREP);

    for (int rep = 0; rep < NREP; rep++) {
        // Limpiar C
        for (int i = 0; i < N*N; i++) C[i] = 0.0f;

        double t0 = omp_get_wtime();
        matmul_omp(A, B, C, N);
        double t1 = omp_get_wtime();

        double ms = (t1 - t0) * 1000.0;
        tiempos[rep] = ms;
        printf("  Rep %d: %.4f ms\n", rep + 1, ms);
    }

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
        C_ref = (float*)malloc(bytes);
        if (C_ref) {
            matmul_serial(A, B, C_ref, N);
            for (int i = 0; i < N*N; i++) {
                double err = fabs((double)C[i] - (double)C_ref[i]);
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
    printf("CSV: OMP_CPU,%d,%.4f,%.4f,%.6e\n", N, t_ms, gflops, max_error);
    printf("============================================\n");

    // ─── Liberar memoria ───
    free(A);
    free(B);
    free(C);

    return 0;
}
