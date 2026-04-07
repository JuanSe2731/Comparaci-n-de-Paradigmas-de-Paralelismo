// matmul_serial.cpp — Multiplicación de matrices serial (referencia)
// Compilar: g++ -O2 -o matmul_serial matmul_serial.cpp
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <chrono>
#include <algorithm>
#include <vector>

// ─── Configuración del protocolo ───
#define NREP 5

// ─── Kernel serial ───
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
    printf("  MULTIPLICACIÓN DE MATRICES — SERIAL\n");
    printf("============================================\n");
    printf("Tamaño de matriz: %d x %d\n", N, N);
    printf("Repeticiones: %d (se reporta la mediana)\n", NREP);
    printf("--------------------------------------------\n");

    // ─── Asignación de memoria ───
    size_t bytes = (size_t)N * N * sizeof(float);
    float* A = (float*)malloc(bytes);
    float* B = (float*)malloc(bytes);
    float* C = (float*)malloc(bytes);

    if (!A || !B || !C) {
        fprintf(stderr, "Error: No se pudo asignar memoria (%zu bytes x 3)\n", bytes);
        return 1;
    }

    // ─── Inicialización determinista ───
    init_matrices(A, B, N);

    // ─── Calentamiento (1 ejecución sin medir) ───
    printf("Ejecutando calentamiento...\n");
    matmul_serial(A, B, C, N);

    // ─── Medición: NREP repeticiones ───
    std::vector<double> tiempos(NREP);
    printf("Ejecutando %d repeticiones...\n", NREP);

    for (int rep = 0; rep < NREP; rep++) {
        // Limpiar C
        for (int i = 0; i < N*N; i++) C[i] = 0.0f;

        auto t0 = std::chrono::high_resolution_clock::now();
        matmul_serial(A, B, C, N);
        auto t1 = std::chrono::high_resolution_clock::now();

        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        tiempos[rep] = ms;
        printf("  Rep %d: %.4f ms\n", rep + 1, ms);
    }

    // ─── Calcular mediana ───
    std::sort(tiempos.begin(), tiempos.end());
    double t_ms = tiempos[NREP / 2];  // mediana

    // ─── Calcular GFLOPS ───
    double gflops = 2.0 * (double)N * N * N / (t_ms * 1e6);

    printf("--------------------------------------------\n");
    printf("Mediana: %.4f ms\n", t_ms);
    printf("Rendimiento: %.4f GFLOPS\n", gflops);

    // ─── Verificación (este ES el serial, sólo reportamos checksum) ───
    double checksum = 0.0;
    float max_val = 0.0f;
    for (int i = 0; i < N*N; i++) {
        checksum += C[i];
        if (fabsf(C[i]) > max_val) max_val = fabsf(C[i]);
    }
    printf("Checksum C: %.6f\n", checksum);
    printf("Max |C[i,j]|: %.6f\n", max_val);
    printf("Error máximo vs serial: 0.000000 (es la referencia)\n");

    // ─── Línea CSV ───
    printf("--------------------------------------------\n");
    printf("CSV: SERIAL,%d,%.4f,%.4f,0.000000\n", N, t_ms, gflops);
    printf("============================================\n");

    // ─── Liberar memoria ───
    free(A);
    free(B);
    free(C);

    return 0;
}
