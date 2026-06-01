"""
TP3 - Inteligencia Artificial - Universidad Siglo 21
Prototipo de Red de Hopfield para identificación de imágenes 10x10 píxeles
Implementa las dos reglas clásicas de aprendizaje: Hebb y Matriz Pseudoinversa

Alumno: Facundo Oliver
"""

import numpy as np

# ============================================================
# 1. DEFINICIÓN DE PATRONES (imágenes 10x10 píxeles, bipolares ±1)
# ============================================================
# Cada patrón representa el "aro" (reemplazado por una figura
# geométrica simple) más una "escuadra" de referencia en la
# esquina inferior izquierda, cuya posición es inalterable.

# Patrón 1: ROMBO (cuadrado rotado 45°) + escuadra
P1 = np.array([
    [-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [-1,-1,-1,-1, 1,-1,-1,-1,-1,-1],
    [-1,-1,-1, 1,-1, 1,-1,-1,-1,-1],
    [-1,-1, 1,-1,-1,-1, 1,-1,-1,-1],
    [-1, 1,-1,-1,-1,-1,-1, 1,-1,-1],
    [-1,-1, 1,-1,-1,-1, 1,-1,-1,-1],
    [-1,-1,-1, 1,-1, 1,-1,-1,-1,-1],
    [-1,-1,-1,-1, 1,-1,-1,-1,-1,-1],
    [ 1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [ 1, 1,-1,-1,-1,-1,-1,-1,-1,-1],
])

# Patrón 2: CRUZ + escuadra
P2 = np.array([
    [-1,-1,-1,-1, 1,-1,-1,-1,-1,-1],
    [-1,-1,-1,-1, 1,-1,-1,-1,-1,-1],
    [-1,-1,-1,-1, 1,-1,-1,-1,-1,-1],
    [-1,-1,-1,-1, 1,-1,-1,-1,-1,-1],
    [-1, 1, 1, 1, 1, 1, 1, 1,-1,-1],
    [-1,-1,-1,-1, 1,-1,-1,-1,-1,-1],
    [-1,-1,-1,-1, 1,-1,-1,-1,-1,-1],
    [-1,-1,-1,-1, 1,-1,-1,-1,-1,-1],
    [ 1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [ 1, 1,-1,-1,-1,-1,-1,-1,-1,-1],
])

# Patrón 3: TRIÁNGULO + escuadra
P3 = np.array([
    [-1,-1,-1,-1, 1,-1,-1,-1,-1,-1],
    [-1,-1,-1, 1, 1, 1,-1,-1,-1,-1],
    [-1,-1,-1, 1,-1, 1,-1,-1,-1,-1],
    [-1,-1, 1,-1,-1,-1, 1,-1,-1,-1],
    [-1,-1, 1,-1,-1,-1, 1,-1,-1,-1],
    [-1, 1,-1,-1,-1,-1,-1, 1,-1,-1],
    [-1, 1, 1, 1, 1, 1, 1, 1,-1,-1],
    [-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [ 1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [ 1, 1,-1,-1,-1,-1,-1,-1,-1,-1],
])

N = 100  # número de neuronas (10x10)


# ============================================================
# 2. VISUALIZACIÓN POR CARACTERES
# ============================================================

def mostrar(patron, titulo=""):
    """Imprime una imagen 10x10 usando caracteres."""
    if titulo:
        print(f"\n{titulo}")
    print("    +" + "-"*21 + "+")
    matriz = patron.reshape(10, 10) if patron.ndim == 1 else patron
    for fila in matriz:
        linea = "    |"
        for p in fila:
            linea += " ▓" if p == 1 else " ·"
        linea += " |"
        print(linea)
    print("    +" + "-"*21 + "+")


# ============================================================
# 3. APRENDIZAJE: Hebb y Pseudoinversa
# ============================================================

def entrenar_hebb(patrones):
    """
    Regla de Hebb:
        W = (1/N) · Σ_k  ξ^k · (ξ^k)^T
    con W_ii = 0 (sin auto-conexiones).
    """
    N = patrones[0].size
    W = np.zeros((N, N))
    for p in patrones:
        v = p.flatten().astype(float)
        W += np.outer(v, v)
    W /= N
    np.fill_diagonal(W, 0)
    return W


def entrenar_pseudoinversa(patrones):
    """
    Matriz pseudoinversa (Personnaz):
        W = X · (X^T · X)^-1 · X^T
    donde X es la matriz cuyas columnas son los patrones.
    """
    N = patrones[0].size
    X = np.column_stack([p.flatten().astype(float) for p in patrones])
    W = X @ np.linalg.pinv(X.T @ X) @ X.T
    np.fill_diagonal(W, 0)
    return W


# ============================================================
# 4. ENERGÍA Y DINÁMICA DE LA RED
# ============================================================

def energia(s, W):
    """Función de energía de Hopfield:  E = -½ · s^T · W · s"""
    return -0.5 * float(s @ W @ s)


def actualizar(s_inicial, W, max_ciclos=20, verbose=True):
    """
    Actualización ASINCRÓNICA: en cada ciclo se recorren todas las
    neuronas en orden aleatorio aplicando:
        s_i(t+1) = sign( Σ_j  w_ij · s_j(t) )
    Itera hasta alcanzar un estado estable o agotar max_ciclos.
    """
    s = s_inicial.flatten().astype(float).copy()
    N = s.size
    energias = [energia(s, W)]
    if verbose:
        print(f"      Energía inicial = {energias[0]:8.2f}")

    for ciclo in range(1, max_ciclos + 1):
        s_anterior = s.copy()
        orden = np.random.permutation(N)
        for i in orden:
            h = float(W[i] @ s)
            s[i] = 1.0 if h >= 0 else -1.0

        e = energia(s, W)
        energias.append(e)
        if verbose:
            print(f"      Ciclo {ciclo:2d}: energía = {e:8.2f}")

        if np.array_equal(s, s_anterior):
            if verbose:
                print(f"      → Convergencia alcanzada en {ciclo} ciclo(s)")
            break

    return s, energias


# ============================================================
# 5. GENERACIÓN DE RUIDO
# ============================================================

def agregar_ruido(patron, porcentaje, semilla=None):
    """Invierte (flip) un porcentaje de píxeles elegidos al azar."""
    if semilla is not None:
        np.random.seed(semilla)
    s = patron.flatten().copy()
    n_ruido = int(round(len(s) * porcentaje))
    indices = np.random.choice(len(s), n_ruido, replace=False)
    s[indices] *= -1
    return s.reshape(patron.shape)


# ============================================================
# 6. EXPERIMENTOS
# ============================================================

def experimento(nombre, W, patron_objetivo, patron_ruidoso):
    print("\n" + "="*55)
    print(f"  EXPERIMENTO: {nombre}")
    print("="*55)

    mostrar(patron_objetivo, "  PATRÓN OBJETIVO (memorizado):")
    mostrar(patron_ruidoso,  "  ENTRADA CON RUIDO:")

    print("\n  Dinámica de la red:")
    s_final, energias = actualizar(patron_ruidoso, W)

    mostrar(s_final, "  SALIDA RECUPERADA:")

    coincidencias = int(np.sum(s_final == patron_objetivo.flatten()))
    porc = 100 * coincidencias / patron_objetivo.size
    print(f"\n  Coincidencia con patrón objetivo: "
          f"{coincidencias}/100 ({porc:.0f}%)")
    print(f"  ΔEnergía: {energias[0]:.2f} → {energias[-1]:.2f}")
    return s_final, energias


# ============================================================
# 7. EJECUCIÓN PRINCIPAL
# ============================================================

if __name__ == "__main__":
    np.random.seed(7)

    print("\n" + "#"*55)
    print("#  TP3 - PROTOTIPO DE RED DE HOPFIELD                #")
    print("#  Identificación de imágenes 10x10 píxeles          #")
    print("#"*55)

    # ---------- ESCENARIO 1: un único patrón ----------
    print("\n\n" + "█"*55)
    print("█  ESCENARIO 1: La red memoriza UN solo patrón")
    print("█"*55)

    patrones_1 = [P1]
    W_hebb_1 = entrenar_hebb(patrones_1)
    W_pinv_1 = entrenar_pseudoinversa(patrones_1)

    # 20% de ruido sobre el rombo
    ruidoso_20 = agregar_ruido(P1, 0.20, semilla=1)

    experimento("HEBB (1 patrón, 20% ruido)",
                W_hebb_1, P1, ruidoso_20)
    experimento("PSEUDOINVERSA (1 patrón, 20% ruido)",
                W_pinv_1, P1, ruidoso_20)

    # ---------- ESCENARIO 2: tres patrones ----------
    print("\n\n" + "█"*55)
    print("█  ESCENARIO 2: La red memoriza TRES patrones")
    print("█  (rombo, cruz y triángulo)")
    print("█"*55)

    patrones_3 = [P1, P2, P3]
    W_hebb_3 = entrenar_hebb(patrones_3)
    W_pinv_3 = entrenar_pseudoinversa(patrones_3)

    # versión ruidosa del rombo (queremos que recupere P1)
    ruidoso_p1 = agregar_ruido(P1, 0.20, semilla=2)

    experimento("HEBB (3 patrones, recupera rombo)",
                W_hebb_3, P1, ruidoso_p1)
    experimento("PSEUDOINVERSA (3 patrones, recupera rombo)",
                W_pinv_3, P1, ruidoso_p1)

    # ---------- ESCENARIO 3: estrés con mucho ruido ----------
    print("\n\n" + "█"*55)
    print("█  ESCENARIO 3: Estrés con 40% de ruido")
    print("█"*55)

    ruidoso_40 = agregar_ruido(P1, 0.40, semilla=3)

    experimento("HEBB (3 patrones, 40% ruido)",
                W_hebb_3, P1, ruidoso_40)
    experimento("PSEUDOINVERSA (3 patrones, 40% ruido)",
                W_pinv_3, P1, ruidoso_40)

    print("\n" + "#"*55)
    print("#  FIN DE LA EJECUCIÓN                              #")
    print("#"*55 + "\n")
