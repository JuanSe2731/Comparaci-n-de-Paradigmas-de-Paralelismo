#!/usr/bin/env python3
# plot_results.py — Genera gráficas comparativas del taller de paralelismo
# Uso: python3 plot_results.py [--csv results.csv] [--output-dir plots]

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
import numpy as np
import os
import sys

# ─── Configuración ───
CSV_FILE = "results.csv"
OUTPUT_DIR = "plots"

# Parseo simple de argumentos
for i, arg in enumerate(sys.argv[1:], 1):
    if arg == "--csv" and i < len(sys.argv) - 1:
        CSV_FILE = sys.argv[i + 1]
    elif arg == "--output-dir" and i < len(sys.argv) - 1:
        OUTPUT_DIR = sys.argv[i + 1]

# ─── Estilo ───
sns.set_theme(style="darkgrid", palette="husl", font_scale=1.1)
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.bbox'] = 'tight'

# ─── Leer datos ───
print(f"Leyendo datos de: {CSV_FILE}")
df = pd.read_csv(CSV_FILE)
print(f"Datos cargados: {len(df)} filas")
print(df.to_string(index=False))
print()

# Crear directorio de salida
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─── Paleta de colores por paradigma ───
COLORES = {
    'SERIAL':      '#636EFA',  # Azul
    'OMP_CPU':     '#00CC96',  # Verde
    'CUDA_NAIVE':  '#EF553B',  # Rojo
    'OMP_GPU':     '#AB63FA',  # Púrpura
    'OMP_GPU_CPU': '#FFA15A',  # Naranja
}

NOMBRES = {
    'SERIAL':      'Serial',
    'OMP_CPU':     'OpenMP CPU',
    'CUDA_NAIVE':  'CUDA Naive',
    'OMP_GPU':     'OpenMP GPU',
    'OMP_GPU_CPU': 'OpenMP GPU (CPU Fallback)',
}

MARCADORES = {
    'SERIAL':      'o',
    'OMP_CPU':     's',
    'CUDA_NAIVE':  '^',
    'OMP_GPU':     'D',
    'OMP_GPU_CPU': 'P',
}

paradigmas = df['paradigma'].unique()

# ═══════════════════════════════════════════
# Gráfica 1: GFLOPS vs N
# ═══════════════════════════════════════════
print("Generando: GFLOPS vs N...")
fig, ax = plt.subplots()

for p in paradigmas:
    subset = df[df['paradigma'] == p].sort_values('N')
    ax.plot(subset['N'], subset['gflops'],
            marker=MARCADORES.get(p, 'o'),
            color=COLORES.get(p, '#333333'),
            label=NOMBRES.get(p, p),
            linewidth=2.5,
            markersize=8)

ax.set_xlabel('Tamaño de Matriz (N)', fontsize=13, fontweight='bold')
ax.set_ylabel('Rendimiento (GFLOPS)', fontsize=13, fontweight='bold')
ax.set_title('Rendimiento (GFLOPS) vs Tamaño de Matriz', fontsize=15, fontweight='bold')
ax.set_xscale('log', base=2)
ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
ax.xaxis.set_major_locator(ticker.FixedLocator(sorted(df['N'].unique())))
ax.legend(fontsize=11, loc='best', framealpha=0.9)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'gflops_vs_N.png'))
plt.close()
print(f"  → {OUTPUT_DIR}/gflops_vs_N.png")

# ═══════════════════════════════════════════
# Gráfica 2: Speedup vs Serial
# ═══════════════════════════════════════════
print("Generando: Speedup vs Serial...")

# Obtener tiempos serial como referencia
serial_times = df[df['paradigma'] == 'SERIAL'][['N', 'tiempo_ms']].rename(
    columns={'tiempo_ms': 'tiempo_serial'})

if not serial_times.empty:
    df_sp = df.merge(serial_times, on='N', how='left')
    df_sp['speedup_vs_serial'] = df_sp['tiempo_serial'] / df_sp['tiempo_ms']

    fig, ax = plt.subplots()

    for p in paradigmas:
        subset = df_sp[df_sp['paradigma'] == p].sort_values('N')
        ax.plot(subset['N'], subset['speedup_vs_serial'],
                marker=MARCADORES.get(p, 'o'),
                color=COLORES.get(p, '#333333'),
                label=NOMBRES.get(p, p),
                linewidth=2.5,
                markersize=8)

    ax.axhline(y=1, color='gray', linestyle='--', linewidth=1, alpha=0.7, label='Baseline (Serial)')
    ax.set_xlabel('Tamaño de Matriz (N)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Speedup vs Serial', fontsize=13, fontweight='bold')
    ax.set_title('Speedup con respecto al Kernel Serial', fontsize=15, fontweight='bold')
    ax.set_xscale('log', base=2)
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
    ax.xaxis.set_major_locator(ticker.FixedLocator(sorted(df['N'].unique())))
    ax.legend(fontsize=11, loc='best', framealpha=0.9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'speedup_vs_serial.png'))
    plt.close()
    print(f"  → {OUTPUT_DIR}/speedup_vs_serial.png")
else:
    print("  ⚠ No hay datos del paradigma SERIAL, omitiendo gráfica.")

# ═══════════════════════════════════════════
# Gráfica 3: Speedup vs OpenMP CPU
# ═══════════════════════════════════════════
print("Generando: Speedup vs OpenMP CPU...")

omp_times = df[df['paradigma'] == 'OMP_CPU'][['N', 'tiempo_ms']].rename(
    columns={'tiempo_ms': 'tiempo_omp_cpu'})

if not omp_times.empty:
    df_sp2 = df.merge(omp_times, on='N', how='left')
    df_sp2['speedup_vs_omp'] = df_sp2['tiempo_omp_cpu'] / df_sp2['tiempo_ms']

    fig, ax = plt.subplots()

    for p in paradigmas:
        subset = df_sp2[df_sp2['paradigma'] == p].sort_values('N')
        ax.plot(subset['N'], subset['speedup_vs_omp'],
                marker=MARCADORES.get(p, 'o'),
                color=COLORES.get(p, '#333333'),
                label=NOMBRES.get(p, p),
                linewidth=2.5,
                markersize=8)

    ax.axhline(y=1, color='gray', linestyle='--', linewidth=1, alpha=0.7, label='Baseline (OpenMP CPU)')
    ax.set_xlabel('Tamaño de Matriz (N)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Speedup vs OpenMP CPU', fontsize=13, fontweight='bold')
    ax.set_title('Speedup con respecto a OpenMP CPU', fontsize=15, fontweight='bold')
    ax.set_xscale('log', base=2)
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
    ax.xaxis.set_major_locator(ticker.FixedLocator(sorted(df['N'].unique())))
    ax.legend(fontsize=11, loc='best', framealpha=0.9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'speedup_vs_omp_cpu.png'))
    plt.close()
    print(f"  → {OUTPUT_DIR}/speedup_vs_omp_cpu.png")
else:
    print("  ⚠ No hay datos del paradigma OMP_CPU, omitiendo gráfica.")

# ═══════════════════════════════════════════
# Gráfica 4: Tiempo de ejecución (Barras)
# ═══════════════════════════════════════════
print("Generando: Tiempo de ejecución (barras)...")

fig, ax = plt.subplots(figsize=(12, 6))

sizes = sorted(df['N'].unique())
x = np.arange(len(sizes))
width = 0.2
n_paradigmas = len(paradigmas)

for i, p in enumerate(paradigmas):
    subset = df[df['paradigma'] == p].sort_values('N')
    offset = (i - n_paradigmas / 2 + 0.5) * width
    bars = ax.bar(x + offset, subset['tiempo_ms'], width,
                  label=NOMBRES.get(p, p),
                  color=COLORES.get(p, '#333333'),
                  alpha=0.85,
                  edgecolor='white',
                  linewidth=0.5)

ax.set_xlabel('Tamaño de Matriz (N)', fontsize=13, fontweight='bold')
ax.set_ylabel('Tiempo (ms)', fontsize=13, fontweight='bold')
ax.set_title('Tiempo de Ejecución por Paradigma', fontsize=15, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels([str(s) for s in sizes])
ax.set_yscale('log')
ax.legend(fontsize=11, loc='best', framealpha=0.9)
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'tiempo_barras.png'))
plt.close()
print(f"  → {OUTPUT_DIR}/tiempo_barras.png")

# ═══════════════════════════════════════════
# Tabla resumen
# ═══════════════════════════════════════════
print("\n" + "=" * 60)
print("  TABLA RESUMEN")
print("=" * 60)
pivot = df.pivot_table(values=['gflops', 'tiempo_ms'], index='N', columns='paradigma')
print(pivot.to_string())
print("=" * 60)

print(f"\n✓ Todas las gráficas guardadas en: {OUTPUT_DIR}/")
