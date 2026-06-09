# Tarea 2: Aprendizaje Distribuido para Computer Vision

| Campo | Detalle |
|---|---|
| **Estudiante** | Esteban Gutiérrez Saborío |
| **Carrera** | Bachillerato en Ingeniería en Ciencia de Datos |
| **Universidad** | Lead University |
| **Curso** | Computación Paralela y Distribuida (TTCT0017) |
| **Profesor** | Johansell Villalobos Cubillo |
| **Fecha de entrega** | 11 de junio, 2026 |

---

## Descripción

Este proyecto implementa un sistema de clasificación de imágenes utilizando **JAX** y **Flax NNX** sobre el dataset *Wonders of the World Image Classification* (12 clases, ~3846 imágenes). Se entrena una red neuronal convolucional (CNN) y se realizan experimentos sistemáticos sobre hiperparámetros, tamaño de red y precisión numérica.

---

## Link de colab

https://colab.research.google.com/drive/1TL8hrN4_XtGq031vOm8HkmIAM66IhqlF?usp=sharing


## Estructura del Proyecto

```
Tarea2_AprendizajeDistribuido_ComputerVision/
├── tarea2_computer_vision_jax.ipynb   # Notebook principal (Google Colab)
├── README.md                          # Este archivo
└── Informe_Tarea2_EstebanGutierrez.docx  # Informe escrito
```

---

## Dataset

**Wonders of the World Image Classification** (Kaggle)

- **Fuente:** https://www.kaggle.com/datasets/
- **Clases (12):** burj_khalifa, chichen_itza, christ_the_reedemer, eiffel_tower, great_wall_of_china, machu_pichu, pyramids_of_giza, roman_colosseum, statue_of_liberty, stonehenge, taj_mahal, venezuela_angel_falls
- **Total de imágenes:** ~3846
- **Resolución de entrada:** 64×64 px (RGB)
- **Split:** 70% train | 15% validación | 15% prueba

---

## Requisitos

### Google Colab (recomendado)

```python
# Ejecutar en la primera celda del notebook
!pip install jax[cuda12] flax optax matplotlib scikit-learn pillow tqdm
```

### Entorno local (macOS Apple Silicon / M-series)

```bash
# Crear entorno virtual
python3 -m venv venv
source venv/bin/activate

# Instalar JAX para Metal (Apple Silicon)
pip install jax-metal

# Instalar dependencias adicionales
pip install flax optax matplotlib scikit-learn pillow tqdm notebook
```

> **Nota:** En macOS M-series, JAX utiliza el backend Metal para aceleración con GPU. Ejecutar `jax.devices()` para confirmar el dispositivo.

---

## Instrucciones de Ejecución

### Google Colab

1. Abrir el notebook `tarea2_computer_vision_jax.ipynb` en Google Colab.
2. Activar GPU: **Runtime → Change Runtime Type → GPU (T4 o A100)**.
3. Subir el dataset al entorno de Colab (ya sea desde Drive o directamente).
4. Ajustar la variable `DATA_DIR` en la celda de configuración del dataset:
   ```python
   DATA_DIR = Path('/content/Wonders of World/Wonders of World')
   ```
5. Ejecutar todas las celdas en orden (**Runtime → Run all**).

### Entorno Local

1. Activar el entorno virtual e iniciar Jupyter:
   ```bash
   source venv/bin/activate
   jupyter notebook
   ```
2. Abrir `tarea2_computer_vision_jax.ipynb`.
3. La variable `DATA_DIR` ya apunta a la ruta local correcta:
   ```python
   DATA_DIR = Path('/Users/esteban07/Desktop/archive/Wonders of World/Wonders of World')
   ```
4. Ejecutar todas las celdas en orden.

---

## Experimentos Incluidos

| Experimento | Variables | Valores |
|---|---|---|
| Tamaño de lote | `batch_size` | 32, 64, 128 |
| Tasa de aprendizaje | `learning_rate` | 1e-2, 1e-3, 1e-4 |
| Tamaño de la red | `filters`, `dense_units` | Pequeña / Mediana / Grande |
| Optimización de HP | Random Search | 8 trials |
| Precisión numérica | `dtype` | float32, float16, bfloat16 |

---

## Arquitectura del Modelo

```
Input (64×64×3)
  ↓
Conv(32) → BatchNorm → ReLU → MaxPool(2×2)
  ↓
Conv(64) → BatchNorm → ReLU → MaxPool(2×2)
  ↓
Conv(128) → BatchNorm → ReLU → MaxPool(2×2)
  ↓
Flatten (8×8×128 = 8192)
  ↓
Dense(256) → ReLU → Dropout(0.4)
  ↓
Dense(12)  ← Logits (salida)
```

---

## Métricas Reportadas

- Exactitud de entrenamiento, validación y prueba
- Curvas de pérdida y exactitud por época
- Tiempo total de entrenamiento
- Tiempo promedio por época
- Throughput (imágenes/segundo)
- Estabilidad numérica (NaN/Inf count)
- Consumo de memoria pico (MB)

---

## Gráficas Generadas

| Archivo | Contenido |
|---|---|
| `samples_dataset.png` | Muestras del dataset por clase |
| `01_baseline_curves.png` | Curvas de entrenamiento base |
| `02_batch_size_experiment.png` | Comparativa de tamaños de lote |
| `03_learning_rate_experiment.png` | Comparativa de tasas de aprendizaje |
| `04_network_size_experiment.png` | Impacto del tamaño de la red |
| `05_hyperparameter_search.png` | Resultados del random search |
| `06_numerical_precision.png` | Comparativa de precisión numérica |
| `06b_precision_loss_curves.png` | Curvas de pérdida por tipo de dato |
| `07_confusion_matrix.png` | Matriz de confusión del mejor modelo |
| `08_throughput_comparison.png` | Comparativa de throughput |

---

## Notas de Reproducibilidad

- Semilla global fija: `SEED = 42`
- Todos los experimentos usan la misma semilla para garantizar resultados reproducibles.
- El notebook registra el tipo de acelerador detectado al inicio.
