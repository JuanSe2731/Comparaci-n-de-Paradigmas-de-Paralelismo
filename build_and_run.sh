#!/bin/bash
# build_and_run.sh — Compila y ejecuta todos los paradigmas de multiplicación de matrices
# Uso: bash build_and_run.sh [--compile-only] [--run-only] [--sizes "512 1024 2048 4096"]
set -e

# ─── Configuración ───
SIZES="${SIZES:-512 1024 2048 4096}"
OUTPUT_CSV="results.csv"
BUILD_DIR="."

# ─── Colores para output ───
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# ─── Parseo de argumentos ───
COMPILE=true
RUN=true

for arg in "$@"; do
    case $arg in
        --compile-only) RUN=false ;;
        --run-only) COMPILE=false ;;
        --sizes) shift; SIZES="$1" ;;
    esac
done

echo -e "${CYAN}════════════════════════════════════════════════════════${NC}"
echo -e "${CYAN}  Taller: Comparación de Paradigmas de Paralelismo${NC}"
echo -e "${CYAN}════════════════════════════════════════════════════════${NC}"
echo ""

# ═══════════════════════════════════════════
#  FASE 1: COMPILACIÓN
# ═══════════════════════════════════════════
if [ "$COMPILE" = true ]; then
    echo -e "${YELLOW}▶ FASE 1: Compilación${NC}"
    echo "────────────────────────────────────────────"

    # 1. Serial
    echo -ne "  [1/5] Compilando matmul_serial.cpp ... "
    if g++ -O2 -o matmul_serial matmul_serial.cpp 2>/dev/null; then
        echo -e "${GREEN}✓${NC}"
    else
        echo -e "${RED}✗ FALLÓ${NC}"
        exit 1
    fi

    # 2. OpenMP CPU
    echo -ne "  [2/5] Compilando matmul_omp.cpp ... "
    if g++ -O2 -fopenmp -o matmul_omp matmul_omp.cpp 2>/dev/null; then
        echo -e "${GREEN}✓${NC}"
    else
        echo -e "${RED}✗ FALLÓ${NC}"
        exit 1
    fi

    # 3. CUDA Naive
    echo -ne "  [3/5] Compilando gemmCudaNaive.cu ... "
    if command -v nvcc &> /dev/null; then
        if nvcc -O2 -o matmul_cuda gemmCudaNaive.cu 2>/dev/null; then
            echo -e "${GREEN}✓${NC}"
        else
            echo -e "${RED}✗ FALLÓ${NC}"
            echo -e "  ${YELLOW}⚠ nvcc encontrado pero la compilación falló${NC}"
        fi
    else
        echo -e "${YELLOW}⚠ nvcc no encontrado — CUDA omitido${NC}"
    fi

    # 4. OpenMP GPU Offload
    echo -ne "  [4/5] Compilando matmul_omp_gpu.cpp (GPU) ... "
    if command -v nvc++ &> /dev/null; then
        # Intentar con NVIDIA HPC SDK (offload real a GPU)
        if nvc++ -O2 -mp=gpu -target=gpu -o matmul_omp_gpu matmul_omp_gpu.cpp 2>/dev/null; then
            echo -e "${GREEN}✓ (nvc++ — GPU offload real)${NC}"
        else
            echo -e "${YELLOW}⚠ nvc++ falló, intentando g++ fallback...${NC}"
            g++ -O2 -fopenmp -o matmul_omp_gpu matmul_omp_gpu.cpp
            echo -e "${GREEN}✓ (g++ — fallback CPU)${NC}"
        fi
    else
        # Fallback: compilar con g++ (ejecutará en CPU)
        if g++ -O2 -fopenmp -o matmul_omp_gpu matmul_omp_gpu.cpp 2>/dev/null; then
            echo -e "${GREEN}✓ (g++ — fallback CPU, sin nvc++)${NC}"
        else
            echo -e "${RED}✗ FALLÓ${NC}"
        fi
    fi

    # 5. OpenMP GPU Offload — CPU fallback (portabilidad)
    echo -ne "  [5/5] Compilando matmul_omp_gpu.cpp (CPU fallback) ... "
    if g++ -O2 -fopenmp -Wl,-rpath,/usr/lib64 -o matmul_omp_gpu_cpu matmul_omp_gpu.cpp 2>/dev/null; then
        echo -e "${GREEN}✓ (g++ — CPU offload, portabilidad)${NC}"
    else
        echo -e "${RED}✗ FALLÓ${NC}"
    fi

    echo ""
    echo -e "${GREEN}Compilación completada.${NC}"
    echo ""
fi

# ═══════════════════════════════════════════
#  FASE 2: EJECUCIÓN Y RECOLECCIÓN
# ═══════════════════════════════════════════
if [ "$RUN" = true ]; then
    echo -e "${YELLOW}▶ FASE 2: Ejecución y recolección de datos${NC}"
    echo "────────────────────────────────────────────"
    echo "Tamaños de matriz: $SIZES"
    echo ""

    # Crear/limpiar archivo CSV
    echo "paradigma,N,tiempo_ms,gflops,max_error" > "$OUTPUT_CSV"

    for N in $SIZES; do
        echo -e "${CYAN}━━━ N = $N ━━━${NC}"

        # Serial
        if [ -f ./matmul_serial ]; then
            echo -e "  ${YELLOW}▸ Serial${NC}"
            OUTPUT=$(./matmul_serial $N 2>&1)
            echo "$OUTPUT"
            CSV_LINE=$(echo "$OUTPUT" | grep "^CSV:" | sed 's/^CSV: //')
            if [ -n "$CSV_LINE" ]; then
                echo "$CSV_LINE" >> "$OUTPUT_CSV"
            fi
            echo ""
        fi

        # OpenMP CPU
        if [ -f ./matmul_omp ]; then
            echo -e "  ${YELLOW}▸ OpenMP CPU${NC}"
            OUTPUT=$(./matmul_omp $N 2>&1)
            echo "$OUTPUT"
            CSV_LINE=$(echo "$OUTPUT" | grep "^CSV:" | sed 's/^CSV: //')
            if [ -n "$CSV_LINE" ]; then
                echo "$CSV_LINE" >> "$OUTPUT_CSV"
            fi
            echo ""
        fi

        # CUDA Naive
        if [ -f ./matmul_cuda ]; then
            echo -e "  ${YELLOW}▸ CUDA Naive${NC}"
            OUTPUT=$(./matmul_cuda $N 2>&1)
            echo "$OUTPUT"
            CSV_LINE=$(echo "$OUTPUT" | grep "^CSV:" | sed 's/^CSV: //')
            if [ -n "$CSV_LINE" ]; then
                echo "$CSV_LINE" >> "$OUTPUT_CSV"
            fi
            echo ""
        fi

        # OpenMP GPU
        if [ -f ./matmul_omp_gpu ]; then
            echo -e "  ${YELLOW}▸ OpenMP GPU Offload${NC}"
            OUTPUT=$(./matmul_omp_gpu $N 2>&1)
            echo "$OUTPUT"
            CSV_LINE=$(echo "$OUTPUT" | grep "^CSV:" | sed 's/^CSV: //')
            if [ -n "$CSV_LINE" ]; then
                echo "$CSV_LINE" >> "$OUTPUT_CSV"
            fi
            echo ""
        fi

        # OpenMP GPU — CPU Fallback (portabilidad)
        if [ -f ./matmul_omp_gpu_cpu ]; then
            echo -e "  ${YELLOW}▸ OpenMP GPU (CPU Fallback)${NC}"
            OUTPUT=$(LD_LIBRARY_PATH=/usr/lib64 ./matmul_omp_gpu_cpu $N 2>&1)
            echo "$OUTPUT"
            # Renombrar paradigma de OMP_GPU a OMP_GPU_CPU en la línea CSV
            CSV_LINE=$(echo "$OUTPUT" | grep "^CSV:" | sed 's/^CSV: //' | sed 's/^OMP_GPU/OMP_GPU_CPU/')
            if [ -n "$CSV_LINE" ]; then
                echo "$CSV_LINE" >> "$OUTPUT_CSV"
            fi
            echo ""
        fi
    done

    echo -e "${CYAN}════════════════════════════════════════════════════════${NC}"
    echo -e "${GREEN}Resultados guardados en: ${OUTPUT_CSV}${NC}"
    echo ""
    echo -e "${YELLOW}Contenido de ${OUTPUT_CSV}:${NC}"
    cat "$OUTPUT_CSV"
    echo ""
    echo -e "${CYAN}════════════════════════════════════════════════════════${NC}"

    # ─── Generar gráficas si Python disponible ───
    if command -v python3 &> /dev/null && [ -f plot_results.py ]; then
        echo -e "${YELLOW}▶ FASE 3: Generando gráficas...${NC}"
        python3 plot_results.py
        echo -e "${GREEN}Gráficas generadas en plots/${NC}"
    else
        echo -e "${YELLOW}⚠ Ejecuta manualmente: python3 plot_results.py${NC}"
    fi
fi
