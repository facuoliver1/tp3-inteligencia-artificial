"""
Genera figuras profesionales del prototipo de Hopfield para incluir en el PDF.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import sys
sys.path.insert(0, '/home/claude')
from hopfield_tp3 import (
    P1, P2, P3,
    entrenar_hebb, entrenar_pseudoinversa,
    actualizar, agregar_ruido, energia
)

# colormap personalizado: blanco y negro con borde sutil
cmap_bn = ListedColormap(['#F5F5F5', '#1a1a1a'])

def plot_imagen(ax, patron, titulo, cmap=cmap_bn):
    """Dibuja un patrón 10x10 en un eje con título."""
    matriz = patron.reshape(10, 10) if patron.ndim == 1 else patron
    ax.imshow(matriz, cmap=cmap, vmin=-1, vmax=1, interpolation='nearest')
    ax.set_title(titulo, fontsize=11, fontweight='bold', pad=8)
    ax.set_xticks([])
    ax.set_yticks([])
    # grilla fina
    ax.set_xticks(np.arange(-0.5, 10, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, 10, 1), minor=True)
    ax.grid(which='minor', color='gray', linewidth=0.3, alpha=0.4)
    for spine in ax.spines.values():
        spine.set_linewidth(1.2)


# ============================================================
# FIGURA 1: Los tres patrones a memorizar
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(10, 3.5))
plot_imagen(axes[0], P1, "Patrón 1: Rombo")
plot_imagen(axes[1], P2, "Patrón 2: Cruz")
plot_imagen(axes[2], P3, "Patrón 3: Triángulo")
fig.suptitle("Patrones objetivo memorizados por la red (10×10 píxeles)",
             fontsize=12, fontweight='bold', y=1.02)
plt.figtext(0.5, -0.02,
            "Cada patrón incluye una escuadra de referencia (esquina inferior izquierda) "
            "cuya posición es inalterable.",
            ha='center', fontsize=9, style='italic', color='#555')
plt.tight_layout()
plt.savefig('/home/claude/fig1_patrones.png', dpi=150, bbox_inches='tight',
            facecolor='white')
plt.close()
print("✓ fig1_patrones.png")


# ============================================================
# FIGURA 2: Recuperación con 1 patrón (Hebb vs Pseudoinversa)
# ============================================================
np.random.seed(7)
W_hebb_1 = entrenar_hebb([P1])
W_pinv_1 = entrenar_pseudoinversa([P1])
ruidoso_20 = agregar_ruido(P1, 0.20, semilla=1)

s_hebb, e_hebb = actualizar(ruidoso_20, W_hebb_1, verbose=False)
s_pinv, e_pinv = actualizar(ruidoso_20, W_pinv_1, verbose=False)

fig, axes = plt.subplots(2, 4, figsize=(13, 6.5))

# fila 1: Hebb
plot_imagen(axes[0,0], P1, "Patrón objetivo")
plot_imagen(axes[0,1], ruidoso_20, "Entrada (20% ruido)")
plot_imagen(axes[0,2], s_hebb, "Salida — Hebb")
axes[0,3].plot(range(len(e_hebb)), e_hebb, 'o-', color='#1f77b4', linewidth=2,
               markersize=8)
axes[0,3].set_title("Energía — Hebb", fontsize=11, fontweight='bold')
axes[0,3].set_xlabel("Ciclo")
axes[0,3].set_ylabel("E")
axes[0,3].grid(alpha=0.3)
axes[0,3].set_xticks(range(len(e_hebb)))

# fila 2: Pseudoinversa
plot_imagen(axes[1,0], P1, "Patrón objetivo")
plot_imagen(axes[1,1], ruidoso_20, "Entrada (20% ruido)")
plot_imagen(axes[1,2], s_pinv, "Salida — Pseudoinversa")
axes[1,3].plot(range(len(e_pinv)), e_pinv, 'o-', color='#d62728', linewidth=2,
               markersize=8)
axes[1,3].set_title("Energía — Pseudoinversa", fontsize=11, fontweight='bold')
axes[1,3].set_xlabel("Ciclo")
axes[1,3].set_ylabel("E")
axes[1,3].grid(alpha=0.3)
axes[1,3].set_xticks(range(len(e_pinv)))

fig.suptitle("Escenario 1: Un único patrón memorizado, 20% de ruido en la entrada",
             fontsize=13, fontweight='bold', y=1.00)
plt.figtext(0.5, -0.01,
            "Ambas reglas recuperan el patrón al 100%. La energía cae monotónicamente "
            "hasta el mínimo del atractor.",
            ha='center', fontsize=9, style='italic', color='#555')
plt.tight_layout()
plt.savefig('/home/claude/fig2_escenario1.png', dpi=150, bbox_inches='tight',
            facecolor='white')
plt.close()
print("✓ fig2_escenario1.png")


# ============================================================
# FIGURA 3: Escenario crítico — 3 patrones, falla de Hebb
# ============================================================
np.random.seed(7)
W_hebb_3 = entrenar_hebb([P1, P2, P3])
W_pinv_3 = entrenar_pseudoinversa([P1, P2, P3])
ruidoso_p1 = agregar_ruido(P1, 0.20, semilla=2)

s_hebb3, e_hebb3 = actualizar(ruidoso_p1, W_hebb_3, verbose=False)
s_pinv3, e_pinv3 = actualizar(ruidoso_p1, W_pinv_3, verbose=False)

fig, axes = plt.subplots(2, 4, figsize=(13, 6.5))

plot_imagen(axes[0,0], P1, "Patrón objetivo (rombo)")
plot_imagen(axes[0,1], ruidoso_p1, "Entrada (20% ruido)")
plot_imagen(axes[0,2], s_hebb3, "Salida — Hebb (FALLA)")
axes[0,3].plot(range(len(e_hebb3)), e_hebb3, 'o-', color='#1f77b4', linewidth=2,
               markersize=8)
axes[0,3].set_title("Energía — Hebb", fontsize=11, fontweight='bold')
axes[0,3].set_xlabel("Ciclo")
axes[0,3].set_ylabel("E")
axes[0,3].grid(alpha=0.3)
axes[0,3].set_xticks(range(len(e_hebb3)))

plot_imagen(axes[1,0], P1, "Patrón objetivo (rombo)")
plot_imagen(axes[1,1], ruidoso_p1, "Entrada (20% ruido)")
plot_imagen(axes[1,2], s_pinv3, "Salida — Pseudoinversa (OK)")
axes[1,3].plot(range(len(e_pinv3)), e_pinv3, 'o-', color='#d62728', linewidth=2,
               markersize=8)
axes[1,3].set_title("Energía — Pseudoinversa", fontsize=11, fontweight='bold')
axes[1,3].set_xlabel("Ciclo")
axes[1,3].set_ylabel("E")
axes[1,3].grid(alpha=0.3)
axes[1,3].set_xticks(range(len(e_pinv3)))

fig.suptitle("Escenario 2: Tres patrones memorizados, 20% de ruido — comparación de reglas",
             fontsize=13, fontweight='bold', y=1.00)
plt.figtext(0.5, -0.01,
            "Hebb converge a un ESTADO ESPURIO (mezcla de los 3 patrones). "
            "Pseudoinversa recupera el rombo al 100%.",
            ha='center', fontsize=9, style='italic', color='#555')
plt.tight_layout()
plt.savefig('/home/claude/fig3_escenario2.png', dpi=150, bbox_inches='tight',
            facecolor='white')
plt.close()
print("✓ fig3_escenario2.png")


# ============================================================
# FIGURA 4: Curva de robustez al ruido (Hebb vs Pseudoinversa)
# ============================================================
np.random.seed(7)
niveles_ruido = np.arange(0, 0.55, 0.05)
n_repeticiones = 20

resultados_hebb = []
resultados_pinv = []

for p in niveles_ruido:
    aciertos_h = []
    aciertos_p = []
    for rep in range(n_repeticiones):
        ruidoso = agregar_ruido(P1, p)
        s_h, _ = actualizar(ruidoso, W_hebb_3, verbose=False)
        s_p, _ = actualizar(ruidoso, W_pinv_3, verbose=False)
        aciertos_h.append(np.mean(s_h == P1.flatten()) * 100)
        aciertos_p.append(np.mean(s_p == P1.flatten()) * 100)
    resultados_hebb.append(np.mean(aciertos_h))
    resultados_pinv.append(np.mean(aciertos_p))

fig, ax = plt.subplots(figsize=(9, 5.5))
ax.plot(niveles_ruido * 100, resultados_hebb, 'o-', color='#1f77b4',
        linewidth=2.5, markersize=9, label='Regla de Hebb')
ax.plot(niveles_ruido * 100, resultados_pinv, 's-', color='#d62728',
        linewidth=2.5, markersize=9, label='Matriz pseudoinversa')
ax.axhline(50, color='gray', linestyle='--', linewidth=1, alpha=0.5,
           label='Azar (50%)')
ax.set_xlabel("Porcentaje de ruido en la entrada (%)", fontsize=11)
ax.set_ylabel("Coincidencia con patrón objetivo (%)", fontsize=11)
ax.set_title("Robustez al ruido: Hebb vs Pseudoinversa\n"
             "(3 patrones memorizados, 20 repeticiones por nivel)",
             fontsize=12, fontweight='bold')
ax.legend(loc='lower left', fontsize=11, framealpha=0.95)
ax.grid(alpha=0.3)
ax.set_ylim(40, 105)
ax.set_xlim(-2, 52)
plt.tight_layout()
plt.savefig('/home/claude/fig4_robustez.png', dpi=150, bbox_inches='tight',
            facecolor='white')
plt.close()
print("✓ fig4_robustez.png")

print("\nTodas las figuras generadas correctamente.")
