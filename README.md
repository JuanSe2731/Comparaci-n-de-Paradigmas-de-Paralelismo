# Comparación de Paradigmas de Paralelismo — GEMM (HPC)

Taller de High-Performance Computing que implementa, mide y compara cinco paradigmas de paralelismo para la **multiplicación de matrices cuadradas** (GEMM, $C = A \times B$) en precisión simple (fp32).

**Grupo 3: OpenMP GPU Offload (`target teams distribute parallel for`)**

---

## Paradigmas Implementados

| Paradigma | Archivo | Compilador | Dispositivo |
|-----------|---------|-----------|-------------|
| Serial | `matmul_serial.cpp` | `g++ -O2` | CPU (1 hilo) |
| OpenMP CPU | `matmul_omp.cpp` | `g++ -O2 -fopenmp` | CPU (multi-hilo) |
| CUDA Naive | `gemmCudaNaive.cu` | `nvcc -O2` | GPU NVIDIA |
| OpenMP GPU | `matmul_omp_gpu.cpp` | `nvc++ -mp=gpu` | GPU (offload) |
| OpenMP GPU (CPU) | `matmul_omp_gpu.cpp` | `g++ -fopenmp` | CPU (fallback portabilidad) |

---

## Requisitos

### Compiladores
- **GCC** (g++) con soporte OpenMP (`-fopenmp`)
- **NVIDIA CUDA Toolkit** (nvcc) para la versión CUDA
- **NVIDIA HPC SDK** (nvc++) para OpenMP GPU Offload real *(opcional — sin nvc++ se compila como CPU fallback)*

### Hardware
- CPU con soporte multi-hilo
- GPU NVIDIA con Compute Capability ≥ 6.0 *(probado en GTX 1650 Ti, CC 7.5)*

### Python (para gráficas)
- Python 3.x
- `matplotlib`, `seaborn`, `pandas`, `numpy`

```bash
pip install matplotlib seaborn pandas numpy
```

---

## Estructura del Proyecto

```
.
├── matmul_serial.cpp       # Implementación serial (referencia)
├── matmul_omp.cpp          # OpenMP CPU (parallel for collapse(2))
├── gemmCudaNaive.cu        # CUDA naive (sin shared memory)
├── matmul_omp_gpu.cpp      # OpenMP GPU Offload (target teams)
├── build_and_run.sh        # Script de compilación y ejecución automática
├── plot_results.py         # Generación de gráficas comparativas
├── presentacion.md         # Presentación Marp (diapositivas)
├── results.csv             # Resultados experimentales
├── plots/                  # Gráficas generadas (PNG)
│   ├── gflops_vs_N.png
│   ├── speedup_vs_serial.png
│   ├── speedup_vs_omp_cpu.png
│   └── tiempo_barras.png
└── README.md
```

---

## Uso Rápido

### Compilar y ejecutar todo automáticamente

```bash
chmod +x build_and_run.sh
bash build_and_run.sh
```

Esto compila los 5 binarios, ejecuta para N ∈ {512, 1024, 2048, 4096}, guarda los resultados en `results.csv` y genera las gráficas en `plots/`.

### Solo compilar

```bash
bash build_and_run.sh --compile-only
```

### Solo ejecutar (si ya están compilados)

```bash
bash build_and_run.sh --run-only
```

### Tamaños personalizados

```bash
SIZES="512 1024" bash build_and_run.sh
```

### Compilación manual de cada binario

```bash
# Serial
g++ -O2 -o matmul_serial matmul_serial.cpp

# OpenMP CPU
g++ -O2 -fopenmp -o matmul_omp matmul_omp.cpp

# CUDA Naive
nvcc -O2 -o matmul_cuda gemmCudaNaive.cu

# OpenMP GPU Offload (GPU real)
nvc++ -O2 -mp=gpu -target=gpu -o matmul_omp_gpu matmul_omp_gpu.cpp

# OpenMP GPU Offload (CPU fallback)
g++ -O2 -fopenmp -Wl,-rpath,/usr/lib64 -o matmul_omp_gpu_cpu matmul_omp_gpu.cpp
```

### Ejecución individual

```bash
./matmul_serial 1024
./matmul_omp 1024
./matmul_cuda 1024
./matmul_omp_gpu 1024
LD_LIBRARY_PATH=/usr/lib64 ./matmul_omp_gpu_cpu 1024
```

### Generar gráficas

```bash
python3 plot_results.py
```

---

## Protocolo de Medición

Todos los binarios implementan un protocolo estandarizado:

1. **Inicialización determinista**: `A[i][j] = (i+j)/N`, `B[i][j] = (i-j+N)/N`
2. **Calentamiento**: 1 ejecución sin medir tiempo
3. **Medición**: 5 repeticiones (NREP=5), se reporta la **mediana** en milisegundos
4. **GFLOPS**: `2.0 × N³ / (t_ms × 10⁶)`
5. **Verificación**: Para N ≤ 1024, comparación contra el kernel serial con tolerancia de error absoluto < 10⁻³

### Formato de salida

Cada binario imprime información verbose y una línea CSV al final:

```
CSV: PARADIGMA,N,tiempo_ms,gflops,max_error
```

---

## Resultados (GTX 1650 Ti)

### Rendimiento (GFLOPS)

| N | Serial | OMP CPU | CUDA Naive | OMP GPU | OMP GPU (CPU) |
|---:|-------:|--------:|-----------:|--------:|--------------:|
| 512 | 0.97 | 4.86 | 203.53 | 109.62 | 5.17 |
| 1024 | 0.32 | 1.33 | 198.80 | 92.00 | 5.11 |
| 2048 | 0.21 | 0.43 | 249.69 | 78.75 | 4.99 |
| 4096 | 0.16 | 0.38 | 251.84 | 82.34 | 4.78 |

### Tiempo de ejecución (ms)

| N | Serial | OMP CPU | CUDA Naive | OMP GPU | OMP GPU (CPU) |
|---:|-------:|--------:|-----------:|--------:|--------------:|
| 512 | 277.30 | 55.20 | 1.32 | 2.45 | 51.91 |
| 1024 | 6,748.36 | 1,619.69 | 10.80 | 23.34 | 420.26 |
| 2048 | 80,663.80 | 39,785.57 | 68.81 | 218.15 | 3,445.55 |
| 4096 | 866,942.64 | 357,704.88 | 545.75 | 1,669.18 | 28,774.18 |

### Hallazgos clave

- **CUDA Naive** logra el mayor rendimiento (~252 GFLOPS para N=4096)
- **OpenMP GPU** alcanza ~82-110 GFLOPS (~40-55% del CUDA naive)
- **Portabilidad demostrada**: el mismo código `matmul_omp_gpu.cpp` compiló y ejecutó tanto en GPU (nvc++) como en CPU (g++) sin modificaciones
- **OMP GPU (CPU Fallback)** mantuvo ~5 GFLOPS estables en todos los tamaños, mientras que OMP CPU cayó a 0.38 GFLOPS en N=4096

---

## Presentación

La presentación está escrita en formato [Marp](https://marp.app/) en `presentacion.md`.

Para visualizarla:

```bash
# Con Marp CLI
npx @marp-team/marp-cli presentacion.md --html

# O con la extensión Marp en VS Code
# Instalar: ext install marp-team.marp-vscode
```

Contenido de la presentación:
- Introducción al algoritmo GEMM
- Conceptos de OpenMP GPU Offload (`target`, `teams`, `distribute`, `map`)
- Equivalencia entre jerarquía OpenMP GPU y CUDA (Grid/Blocks/Threads)
- Código implementado y explicación
- Resultados experimentales con gráficas y tablas
- Análisis comparativo OpenMP GPU vs CUDA Naive
- Conclusiones

---

## Nota sobre libgomp (NVIDIA HPC SDK)

Si tienes el NVIDIA HPC SDK instalado, su `libgomp.so` puede causar conflictos con la de GCC al ejecutar la versión CPU fallback:

```
symbol lookup error: undefined symbol: GOMP_teams4, version GOMP_5.1
```

**Solución**: compilar con `-Wl,-rpath,/usr/lib64` y ejecutar con `LD_LIBRARY_PATH=/usr/lib64`:

```bash
g++ -O2 -fopenmp -Wl,-rpath,/usr/lib64 -o matmul_omp_gpu_cpu matmul_omp_gpu.cpp
LD_LIBRARY_PATH=/usr/lib64 ./matmul_omp_gpu_cpu 1024
```

---

## Licencia

Proyecto académico — Taller de Comparación de Paradigmas de Paralelismo (HPC).
