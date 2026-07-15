# TFG — Restauración interactiva de señales de audio degradadas mediante aprendizaje profundo

TFG del Grado en Ingeniería de Telecomunicaciones (ETSIT, Universitat Politècnica de València).

**Título completo:** *Análisis y aplicación de modelos de aprendizaje profundo para la restauración interactiva de señales de audio degradadas.*

**Autor:** Rodri — Tutor: alalbiol

## Descripción

El proyecto aborda cinco categorías de degradación de audio — eliminación de ruido (*denoising*), eliminación de reverberación (*dereverberation*), super-resolución o ampliación de ancho de banda (*BWE*), corrección de recorte (*de-clipping*) y separación de fuentes — mediante la aplicación de modelos de aprendizaje profundo **preentrenados** de Hugging Face, sin entrenamiento propio.

Para cada categoría se emplea un modelo especializado, más un modelo combinado multi-tarea y un baseline clásico no basado en IA, comparados mediante métricas bibliográficas (PESQ, STOI, SI-SDR, LSD) y métricas empíricas no intrusivas (DNSMOS, NISQA) sobre audio real sin referencia limpia.

El desarrollo técnico se realiza en Google Colab (GPU) y el resultado final se integra en una aplicación interactiva con Gradio, desplegada como Hugging Face Space.

## Modelos utilizados

| Categoría | Modelo |
|---|---|
| Denoising | DeepFilterNet2/3 |
| Dereverberation | MP-SENet |
| Super-resolución (BWE) | AudioSR |
| De-clipping | VoiceFixer v2 |
| Separación de fuentes | HTDemucs v4 |
| Combinado multi-tarea | ClearVoice / MossFormer2 |

Cada notebook incluye además un baseline clásico (no basado en IA) como referencia comparativa.

## Estructura del repositorio

```
TFG_restauracion_audio/
├── README.md
├── .gitignore
├── GIT_CHEATSHEET.md               # Comandos git de referencia rápida
├── context/                        # Seguimiento del proyecto (para el tutor)
│   ├── objectives.md                # Objetivos del TFG
│   ├── TFG_plan.md                  # Plan de fases y milestones
│   └── diary.md                     # Diario de avances
├── utils/                          # Utilidades compartidas (Python)
│   ├── audio_utils_funcionescomunes.py
│   ├── audio_utils_baselines_clasicos.py
│   ├── audio_utils_metricas_ref.py
│   ├── audio_utils_metricas_no_intrusivas.py
│   └── audio_utils_memoria_GPU.py
├── 00_Setup_Base.ipynb
├── 01_DeepFilterNet_3.ipynb
├── 02_HTDemucs.ipynb
├── 03_VoiceFixer.ipynb
├── 04_MPSENet.ipynb
├── 05_AudioSR.ipynb
└── 06_ClearVoice_MossFormer2.ipynb
```

> Las carpetas `outputs/` (audios procesados) y `cache/` (pesos de modelos descargados) se generan localmente al ejecutar los notebooks y están excluidas del repositorio (ver `.gitignore`) por su tamaño.

## Estado actual del proyecto

Seguimiento detallado, hitos y fechas en [`context/TFG_plan.md`](context/TFG_plan.md).

## Cómo ejecutar este proyecto (Google Colab)

Estos notebooks están pensados para ejecutarse en **Google Colab con GPU**. No dependen de una ruta de Drive concreta: basta con clonar el repositorio en cualquier carpeta de tu Google Drive y ajustar una única variable al principio de cada notebook.

### 1. Clonar el repositorio en tu Google Drive

Puedes hacerlo de dos formas:

**a) Desde tu ordenador**, clonando en local y subiendo la carpeta a Drive:
```bash
git clone https://github.com/dehonidas9/TFG_restauracion_audio.git
```

**b) Directamente desde una celda de Colab**, con Drive ya montado:
```python
from google.colab import drive
drive.mount('/content/drive')

%cd /content/drive/MyDrive
!git clone https://github.com/dehonidas9/TFG_restauracion_audio.git
```

### 2. Configurar la ruta del proyecto en el notebook

Cada notebook empieza con una celda de configuración. Solo hay que ajustar `PROJECT_ROOT` a la ruta donde hayas clonado el repositorio dentro de tu Drive:

```python
from google.colab import drive
drive.mount('/content/drive')

PROJECT_ROOT = '/content/drive/MyDrive/TFG_restauracion_audio'  # <- ajustar si lo clonaste en otra ubicación
%cd {PROJECT_ROOT}

import sys
sys.path.append(PROJECT_ROOT)
```

A partir de aquí, el resto del notebook usa rutas relativas a `PROJECT_ROOT` (por ejemplo, `f"{PROJECT_ROOT}/utils"` o `f"{PROJECT_ROOT}/outputs"`), por lo que funciona igual sin importar en qué cuenta de Drive o carpeta esté clonado.

### 3. Instalar dependencias y ejecutar

Cada notebook instala sus propias dependencias en la primera celda (distintas según el modelo). Basta con ejecutar las celdas en orden; el patrón general es:

1. Instalación de dependencias específicas del modelo
2. Importación de utilidades compartidas (`utils/`)
3. Selección del audio de prueba (se pide por `input()`, listando los archivos disponibles)
4. Inferencia con el modelo
5. Guardado de resultados en `outputs/` (se crea automáticamente, no está en el repo)
6. Cálculo de métricas (DNSMOS y, si aplica, baseline clásico)
7. Liberación de memoria GPU

### Requisitos

- Cuenta de Google con acceso a Google Colab y Google Drive
- Runtime de Colab con GPU (T4 es suficiente)
- Para el bloque empírico: audio real de prueba (no incluido en el repo por privacidad/tamaño; contactar con el autor si es necesario)

## Metodología de evaluación

- **Bloque bibliográfico:** métricas ya publicadas por cada modelo (PESQ, STOI, SI-SDR, LSD) sobre datasets académicos de referencia (VoiceBank+DEMAND, MUSDB18, DNS Challenge), usados como referencia del estado del arte, no como parte del bloque empírico propio.
- **Bloque empírico:** audio real (grabación con reverberación, sin referencia limpia), evaluado con métricas no intrusivas (DNSMOS, NISQA).
- **Comparativas cruzadas:** modelo individual vs. combinado, IA vs. baseline clásico, bloque bibliográfico vs. bloque empírico.

## Documentación adicional

- [`context/objectives.md`](context/objectives.md) — objetivos del TFG
- [`context/TFG_plan.md`](context/TFG_plan.md) — plan de fases y milestones
- [`context/diary.md`](context/diary.md) — diario de seguimiento
- [`GIT_CHEATSHEET.md`](GIT_CHEATSHEET.md) — comandos git de referencia
