# IA-AsBuilt

## Detección de Líneas Estructurales — Pipeline de Entrenamiento Progresivo para Segmentación

Pipeline de segmentación basado en Deep Learning entrenado progresivamente con datasets sintéticos para detectar trazas estructurales en planos técnicos y esquemas de tuberías industriales.

---

# Descripción General

Este proyecto explora técnicas de segmentación semántica para la detección de líneas estructurales mediante datasets sintéticos progresivamente más complejos.

El entrenamiento evoluciona en múltiples fases:

1. Geometría básica
2. Geometría compleja
3. Topologías industriales de tuberías con ruido y oclusiones

El objetivo es desarrollar un modelo robusto capaz de identificar trazas estructurales para aplicaciones de documentación **As-Built**.

---

# Dataset y Modelos: 

Dataset: https://huggingface.co/leuss1224/asbuilt-unet-models

Modelos: https://huggingface.co/datasets/leuss1224/asbuilt-unet-datasets

---

# Entorno de Desarrollo

| Componente | Detalle      |
| ---------- | ------------ |
| Plataforma | Google Colab |
| GPU        | NVIDIA T4    |
| Lenguaje   | Python       |
| Framework  | PyTorch      |

---

# Dependencias

```python id="4o1yqs"
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import torchvision.transforms as transforms

import numpy as np
from PIL import Image
import os
import json
import random
import shutil
from tqdm import tqdm
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
```

---

# Estructura del Dataset

```text id="4j3u8e"
Mi unidad/
└── trazos/
    ├── dataset/
    │   ├── fase1/
    │   │   ├── ruido/
    │   │   │   ├── train/  (inputs/ masks/)
    │   │   │   ├── val/    (inputs/ masks/)
    │   │   │   └── test/   (inputs/ masks/)
    │   │   └── sinteticas/
    │   │       ├── train/  (inputs/ masks/)
    │   │       ├── val/    (inputs/ masks/)
    │   │       └── test/   (inputs/ masks/)
    │   ├── fase2/
    │   │   └── (misma estructura)
    │   └── fase3/
    │       └── (misma estructura)
    ├── modelos/
    │   ├── fase1/
    │   ├── fase2/
    │   └── fase3/
    └── logs/
        ├── fase1/
        ├── fase2/
        └── fase3/
```

---

# Parámetros del Dataset

| Parámetro            | Valor                           | Propósito                                   |
| -------------------- | ------------------------------- | ------------------------------------------- |
| Resolución de imagen | 256 × 256 px                    | Ligero y eficiente para Colab               |
| Formato              | PNG                             | Formato sin pérdida para preservar máscaras |
| División             | 1000 train / 200 val / 200 test | ~1400 imágenes por fase                     |
| Tamaño por fase      | ~50–80 MB                       | Fácil de manejar en Drive                   |
| Tamaño total         | ~250–300 MB                     | Adecuado para Colab                         |

---

# Fases del Dataset

| Fase   | Descripción                            |
| ------ | -------------------------------------- |
| Fase 1 | Geometría básica + ruido sintético     |
| Fase 2 | Geometría compleja + ruido sintético   |
| Fase 3 | Tuberías industriales + ruido realista |

---

# Versiones de Entrenamiento

Cada versión está implementada como un notebook independiente utilizando la misma base tecnológica y pipeline de entrenamiento.

---

# v1 — CNN Base

## Configuración

| Parámetro    | Valor                                     |
| ------------ | ----------------------------------------- |
| Dataset      | Fase 1                                    |
| Arquitectura | CNN simple                                |
| Objetivo     | Establecer una línea base de segmentación |

La primera versión construye el pipeline completo de segmentación y sirve como referencia inicial antes de introducir arquitecturas encoder-decoder.

También se utiliza el detector clásico de bordes Canny como benchmark no neuronal.

---

## Pipeline del Notebook

| Celda | Función                                   |
| ----- | ----------------------------------------- |
| 1     | Librerías                                 |
| 2     | Montaje de Drive y configuración de rutas |
| 3     | Clase Dataset personalizada               |
| 4     | Validación del dataset                    |
| 5     | Inspección visual de overlays             |
| 6     | DataLoaders                               |
| 7     | Baseline Canny                            |
| 8     | Arquitectura CNN                          |
| 9     | Configuración de entrenamiento            |
| 10    | Métricas (IoU / Dice)                     |
| 11    | Loop de entrenamiento + EarlyStopping     |
| 12    | Curvas de aprendizaje                     |
| 13    | Evaluación en Test                        |
| 14    | Visualización de predicciones             |
| 15    | Análisis FP/FN                            |
| 16    | Resumen de resultados                     |
| 17    | Pipeline de inferencia                    |
| 18    | Notas de versión                          |

---

## Resultados

| Métrica            | Resultado |
| ------------------ | --------- |
| IoU Baseline Canny | 0.0555    |
| Test IoU           | 0.5833    |
| Test Dice          | 0.7363    |
| Mejora sobre Canny | ~10×      |

---

## Análisis de Errores

| Muestra | FP  | FN  | Observación                         |
| ------- | --- | --- | ----------------------------------- |
| 1       | 122 | 96  | Ligero desplazamiento posicional    |
| 2       | 169 | 137 | Desplazamiento en líneas largas     |
| 3       | 251 | 165 | Intersecciones generan ambigüedad   |
| 4       | 276 | 88  | Buena detección pero bordes gruesos |

---

# v2 — U-Net

## Configuración

| Parámetro    | Valor                               |
| ------------ | ----------------------------------- |
| Dataset      | Fase 1                              |
| Arquitectura | U-Net                               |
| Objetivo     | Recuperar detalles espaciales finos |

Las conexiones skip del encoder-decoder permiten reconstruir detalles espaciales de alta frecuencia como:

* Esquinas definidas
* Líneas continuas delgadas
* Intersecciones limpias
* Bordes precisos

Esto mejora significativamente la calidad de segmentación respecto a la CNN base.

---

## Resultados

| Métrica | v1 CNN | v2 U-Net | Mejora  |
| ------- | ------ | -------- | ------- |
| IoU     | 0.5833 | 0.9299   | +0.3466 |
| Dice    | 0.7363 | 0.9635   | +0.2272 |

---

## Análisis de Errores

| Muestra | FP | FN | Comparación con v1 |
| ------- | -- | -- | ------------------ |
| 1       | 0  | 0  | v1: 122 / 96       |
| 2       | 48 | 86 | v1: 169 / 137      |
| 3       | 0  | 0  | v1: 251 / 165      |
| 4       | 0  | 0  | v1: 276 / 88       |

---

# v3 — U-Net Progresivo (Fase 1 + Fase 2)

## Configuración

| Parámetro    | Valor                                                       |
| ------------ | ----------------------------------------------------------- |
| Dataset      | Fase 1 + Fase 2                                             |
| Arquitectura | U-Net                                                       |
| Objetivo     | Aprender geometría compleja sin olvidar conocimiento previo |

El modelo se entrena progresivamente con estructuras más difíciles manteniendo el conocimiento adquirido en fases anteriores.

Se reduce el learning rate para estabilizar el entrenamiento en geometrías complejas.

---

## Resultados

| Métrica | v2     | v3     |
| ------- | ------ | ------ |
| IoU     | 0.9299 | 0.9329 |
| Dice    | 0.9635 | 0.9651 |

---

## Resultados por Dataset

| Dataset                     | IoU    | Dice   |
| --------------------------- | ------ | ------ |
| Fase 1 — Geometría Básica   | 0.9390 | 0.9685 |
| Fase 1 — Ruido              | 0.9206 | 0.9585 |
| Fase 2 — Geometría Compleja | 0.9517 | 0.9752 |
| Fase 2 — Ruido Complejo     | 0.9201 | 0.9583 |

---

## Observaciones Clave

* No se detectó catastrophic forgetting
* Rendimiento estable en todos los datasets
* Alta robustez frente a ruido sintético
* Los errores principales ocurren en intersecciones y arcos

---

# v4 — U-Net Progresivo (Fase 1 + Fase 2 + Fase 3)

## Configuración

| Parámetro    | Valor                                  |
| ------------ | -------------------------------------- |
| Dataset      | Fase 1 + Fase 2 + Fase 3               |
| Arquitectura | U-Net                                  |
| Objetivo     | Comprensión de topologías industriales |

La Fase 3 introduce escenarios industriales como:

* Conexiones tipo T
* Cruces de tuberías
* Abrazaderas
* Sombras
* Representación de ejes centrales

También se implementó una optimización importante:

El dataset completo se copia desde Google Drive hacia la RAM de Colab antes del entrenamiento, reduciendo el tiempo de carga inicial de varias horas a pocos minutos.

---

## Resultados

| Métrica              | v2     | v3     | v4     |
| -------------------- | ------ | ------ | ------ |
| IoU Global           | 0.9299 | 0.9329 | 0.9303 |
| Dice Global          | 0.9635 | 0.9651 | 0.9637 |
| IoU Tuberías Fase 3  | —      | —      | 0.9277 |
| Dice Tuberías Fase 3 | —      | —      | 0.9624 |

---

## Resultados por Dataset

| Dataset                      | IoU    | Estado                  |
| ---------------------------- | ------ | ----------------------- |
| Fase 1 — Geometría Básica    | 0.9372 | Estable                 |
| Fase 1 — Ruido               | 0.9219 | Robusto                 |
| Fase 2 — Geometría Compleja  | 0.9511 | Mejor rendimiento       |
| Fase 2 — Ruido Complejo      | 0.9159 | Estable                 |
| Fase 3 — Tuberías Sintéticas | 0.9296 | Aprendido correctamente |
| Fase 3 — Tuberías + Ruido    | 0.9259 | Robusto                 |

---

## Hallazgos Principales

* La Fase 3 superó el objetivo de IoU = 0.85
* El modelo ignora correctamente sombras y abrazaderas
* Las conexiones tipo T y cruces se segmentan correctamente
* No hubo catastrophic forgetting

La ligera caída global respecto a v3 sugiere un leve overfitting después de la época 25.

EarlyStopping se activó en la época 37.

---

# Prueba de Inferencia en Escenarios Reales

El modelo v4 fue evaluado con fotografías reales de instalaciones PVC utilizando múltiples thresholds (0.3 / 0.5 / 0.7).

---

## Observaciones

| Imagen       | Detección    | Problema                                        |
| ------------ | ------------ | ----------------------------------------------- |
| tuberia.png  | 1.4% píxeles | Contraste extremadamente bajo                   |
| tuberia7.png | 1.6% píxeles | Regiones correctas pero fragmentadas            |
| tuberia6.png | 6.7% píxeles | Patrones de baldosas confundidos con estructura |

---

## Casos Exitosos

* El modelo detecta parcialmente regiones estructurales
* No aparece ruido saturado de segmentación
* Buena localización en varios escenarios reales

---

## Casos Fallidos

* Trazas fragmentadas
* Confusión con texturas tipo rejilla o ladrillo
* Bajo contraste tubería-fondo
* Tuberías curvas ausentes en el entrenamiento sintético

---

# Brecha Synthetic-to-Real

| Entrenamiento Sintético  | Imágenes Reales            |
| ------------------------ | -------------------------- |
| Fondos uniformes         | Entornos complejos         |
| Alto contraste           | Bajo contraste             |
| Geometría perfecta       | Distorsión por perspectiva |
| Sin patrones repetitivos | Baldosas y rejillas        |

---

# Roadmap — v5

Mejoras planificadas:

* Añadir fondos de ladrillo, baldosas y rejillas al generador sintético
* Introducir combinaciones de bajo contraste tubería-fondo
* Agregar curvaturas suaves y perspectiva
* Reentrenar con mayor diversidad sintética
* Reevaluar usando las mismas imágenes reales

---

# Referencia de Métricas

| Métrica             | Descripción                                          |
| ------------------- | ---------------------------------------------------- |
| IoU                 | Mide el solapamiento entre predicción y ground truth |
| Dice Score          | Media armónica entre precisión y recall              |
| FP (False Positive) | Ruido detectado como estructura                      |
| FN (False Negative) | Estructura no detectada                              |

En aplicaciones As-Built, los **False Negatives son el error más crítico**, ya que omitir una tubería es peor que detectar ruido adicional.

---

# Trabajo Futuro

Posibles líneas de investigación futuras:

* Attention U-Net
* DeepLabV3+
* Domain Adaptation
* Transfer Learning Synthetic-to-Real
* Generación sintética mediante Diffusion Models
* Vision Transformers para segmentación estructural

---

# Licencia

Proyecto desarrollado con fines educativos y de investigación.
