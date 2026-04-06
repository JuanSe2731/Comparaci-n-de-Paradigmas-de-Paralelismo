/*// matmul_omp_gpu.cpp — compilar:
// NVIDIA: nvc++ -O2 -mp=gpu -target=gpu -o matmul_omp_gpu matmul_omp_gpu.cpp
// AMD: amdclang++ -O2 -fopenmp --offload-arch=gfx90a -o ...
// GCC fallback (sin GPU): g++ -O2 -fopenmp -o matmul_omp_gpu matmul_omp_gpu.cpp
#include <omp.h>
#include <cstdio>
#include <iostream>
#include <ostream>

void matmul_omp_gpu(const float* A, const float* B, float* C, int N) {
 	#pragma omp target data map(to:A[0:N*N], B[0:N*N]) map(from:C[0:N*N])
 	{
 		#pragma omp target teams distribute parallel for collapse(2) \
 			map(to:A[0:N*N],B[0:N*N]) map(from:C[0:N*N])
 		for (int i = 0; i < N; i++)
 			for (int j = 0; j < N; j++) {
 				float acc = 0.0f;
 				for (int k = 0; k < N; k++)
					acc += A[i*N + k] * B[k*N + j];
 					C[i*N + j] = acc;
 				}
 			}
		}

int main() {
	const int N = 1024;
	float *A = new float[N*N], *B = new float[N*N], *C = new float[N*N];
	// TODO: inicializar, medir con omp_get_wtime()
	// Verificar con: omp_get_default_device() y omp_is_initial_device()

	double timestart = omp_get_wtime();

	int default_device = omp_get_default_device();
#pragma omp target device(default_device)
	{
		if (omp_is_initial_device()) {
			printf("Target: Ejecutando en el HOST (Fallback)\n");
		} else {
			printf("Target: Ejecutando en la GPU\n");
		}
		matmul_omp_gpu(A, B, C, N);
	}

	double timeend = omp_get_wtime();
	std::cout << "Time elapsed: " << timeend - timestart << std::endl;

}*/


#include <omp.h>
#include <cstdio>
#include <iostream>
#include <vector>

// 1. Declaramos la función para que el compilador sepa que debe generar código de GPU para ella
#pragma omp declare target
void matmul_omp_gpu(const float* A, const float* B, float* C, int N) {
	// Usamos 'target teams distribute parallel for' para paralelizar en la GPU
#pragma omp target teams distribute parallel for collapse(2) map(to:A[0:N*N], B[0:N*N]) map(from:C[0:N*N])
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
#pragma omp end declare target

int main() {
	const int N = 1024;
	// Usar vectores o inicializar memoria para evitar basura en los cálculos
	std::vector<float> A(N*N, 1.0f);
	std::vector<float> B(N*N, 2.0f);
	std::vector<float> C(N*N, 0.0f);

	// Verificar dispositivo antes de llamar a la función
	int is_cpu = 1;
#pragma omp target map(from:is_cpu)
	{
		is_cpu = omp_is_initial_device();
	}

	if (is_cpu) {
		printf("Corriendo en: HOST (CPU) - Revisa tu instalación de drivers/SDK\n");
	} else {
		printf("Corriendo en: DEVICE (GPU)\n");
	}

	double timestart = omp_get_wtime();

	// Llamada directa: la función ya tiene las directivas de movimiento de datos (map)
	matmul_omp_gpu(A.data(), B.data(), C.data(), N);

	double timeend = omp_get_wtime();
	std::cout << "Time elapsed: " << timeend - timestart << " segundos" << std::endl;

	return 0;
}