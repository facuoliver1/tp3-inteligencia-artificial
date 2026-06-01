# TP3 - Inteligencia Artificial

**Universidad Siglo 21 — Licenciatura en Informática**
**Alumno:** Facundo Oliver

Prototipo de **Red de Hopfield** para identificación de imágenes 10×10 píxeles, implementando las dos reglas clásicas de aprendizaje: **Hebb** y **Matriz Pseudoinversa**.

## Contexto del problema

En una línea de montaje robotizada, ocasionales desplazamientos del *block* del motor sobre la cinta transportadora impiden el posicionamiento correcto de una pieza. Este TP aborda el problema desde el reconocimiento de imágenes: la red de Hopfield "limpia" una imagen ruidosa de la cara lateral del motor para que el robot pueda identificar el aro y calcular las coordenadas del punto de montaje.

## Estructura del repositorio

- `hopfield_tp3.py` — Implementación principal del prototipo. Define los patrones, entrena la red con ambas reglas, ejecuta los tres escenarios experimentales.
- `generar_figuras.py` — Script auxiliar que produce las cuatro figuras del reporte usando matplotlib.
- `fig1_patrones.png` — Los tres patrones objetivo (rombo, cruz, triángulo).
- `fig2_escenario1.png` — Recuperación de un único patrón con 20% de ruido.
- `fig3_escenario2.png` — Tres patrones memorizados: Hebb falla (estado espurio) vs Pseudoinversa recupera al 100%.
- `fig4_robustez.png` — Curva de aciertos vs nivel de ruido.

## Cómo ejecutar

Requisitos: Python 3.8+ con NumPy y Matplotlib.

```bash
pip install numpy matplotlib
python hopfield_tp3.py        # corre los experimentos en consola
python generar_figuras.py     # regenera las figuras
```

## Resultados principales

| Escenario | Hebb | Pseudoinversa |
|-----------|:----:|:-------------:|
| 1 patrón memorizado, 20% ruido | 100% | 100% |
| 3 patrones memorizados, 20% ruido | 94% (estado espurio) | 100% |
| 3 patrones memorizados, 40% ruido | 94% (estado espurio) | 100% |

La regla de Hebb queda atrapada en un mínimo de energía espurio cuando los patrones almacenados están correlacionados (en este caso, comparten la escuadra de referencia). La matriz pseudoinversa es robusta al ruido hasta aproximadamente el 35% antes de degradarse.

## Referencias

- Hopfield, J. J. (1982). *Neural networks and physical systems with emergent collective computational abilities*. PNAS, 79(8), 2554–2558.
- Russell, S., & Norvig, P. (2010). *Artificial Intelligence: A Modern Approach* (3rd ed.). Pearson.
- Personnaz, L., Guyon, I., & Dreyfus, G. (1986). *Collective computational properties of neural networks: New learning mechanisms*. Physical Review A, 34(5), 4217.
