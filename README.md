 # TFG — Restauración interactiva de señales de audio degradadas mediante aprendizaje profundo

TFG del Grado en Ingeniería de Telecomunicaciones (ETSIT, Universitat Politècnica de València).

**Título completo:** *Análisis y aplicación de modelos de aprendizaje profundo para la restauración interactiva de señales de audio degradadas.*

## Descripción

El proyecto aborda cinco categorías de degradación de audio — eliminación de ruido (*denoising*), eliminación de reverberación (*dereverberation*), super-resolución o ampliación de ancho de banda (*BWE*), corrección de recorte (*de-clipping*) y separación de fuentes — mediante la aplicación de modelos de aprendizaje profundo **preentrenados** de Hugging Face, sin entrenamiento propio.

Para cada categoría se emplea un modelo especializado, un modelo combinado multi-tarea y un baseline clásico no basado en IA, comparados mediante métricas bibliográficas (PESQ, STOI, SI-SDR, LSD) y métricas empíricas no intrusivas (DNSMOS, NISQA) sobre audio real sin referencia limpia.

El desarrollo técnico se realiza en Google Colab (GPU T4) y el resultado final se integra en una aplicación interactiva con Gradio, desplegada como Hugging Face Space.

## Modelos utilizados

| Categoría | Modelo | Baseline clásico |
|---|---|---|
| Denoising | DeepFilterNet2/3 | *(clásico, ver notebook)* |
| Dereverberation | MP-SENet | *(clásico, ver notebook)* |
| Super-resolución (BWE) | AudioSR | *(clásico, ver notebook)* |
| De-clipping | VoiceFixer v2 | *(clásico, ver notebook)* |
| Separación de fuentes | HTDemucs v4 | *(clásico, ver notebook)* |
| Combinado multi-tarea | ClearVoice / MossFormer2 | — |

## Estructura del repositorio

```
Proyecto_Audio/
├── context/                        # Seguimiento del proyecto
│   ├── objectives.md                # Objetivos del TFG
│   ├── TFG_plan.md                  # Plan de fases y milestones
│   └── diary.md                     # Diario de avances
├── utils/                          # Utilidades compartidas (.py)
│   ├── audio_utils_baselines_clasicos.py
│   ├── audio_utils_metricas_ref.py
│   ├── audio_utils_metricas_no_intrusivas.py
│   └── audio_utils_memoria_GPU.py
├── notebooks/
│   ├── 00_Setup_Base.ipynb          # ✅ Completado
│   ├── 01_DeepFilterNet_3.ipynb     # ✅ Completado
│   ├── 02_HTDemucs.ipynb            # ⏳ En progreso
│   ├── 03_VoiceFixer.ipynb          # 🔲 Pendiente
│   ├── 04_MPSENet.ipynb             # 🔲 Pendiente
│   ├── 05_AudioSR.ipynb             # 🔲 Pendiente
│   └── 06_ClearVoice_MossFormer2.ipynb  # 🔲 Pendiente
├── outputs/                        # Audios procesados de prueba
├── NOTAS_DESPLIEGUE_FASE5.md        # Notas de despliegue a HF Space
└── README.md
```

> Nota: la estructura exacta de `notebooks/` y `outputs/` se irá completando a medida que avancen las fases; ver detalle en `context/TFG_plan.md`.

## Estado actual del proyecto

- ✅ Fase 0 — Configuración base (Drive, utilidades, métricas, gestión de memoria GPU)
- ✅ Notebook DeepFilterNet2/3 (denoising) finalizado y depurado
- ⏳ Notebook HTDemucs v4 (separación de fuentes) — en curso
- 🔲 Notebooks restantes: VoiceFixer v2, MP-SENet, AudioSR, ClearVoice/MossFormer2
- 🔲 Recopilación bibliográfica de métricas
- 🔲 Interfaz Gradio
- 🔲 Despliegue en Hugging Face Space

Seguimiento detallado, hitos y fechas en [`context/TFG_plan.md`](context/TFG_plan.md).

## Cómo ejecutar

1. Montar Google Drive en Colab y situarse en la carpeta del proyecto:

```python
from google.colab import drive
drive.mount('/content/drive')
%cd /content/drive/MyDrive/Proyecto_Audio
```

2. Abrir el notebook correspondiente a la fase en curso (ver tabla de arriba) y ejecutar las celdas en orden. Cada notebook sigue el mismo patrón:
   - Instalación de dependencias específicas del modelo
   - Importación de utilidades compartidas (`utils/`)
   - Inferencia sobre audio de prueba
   - Guardado de resultados en `outputs/`
   - Cálculo de DNSMOS
   - Ejecución del baseline clásico equivalente
   - Liberación de memoria GPU

3. Requiere GPU (T4 en Colab gratuito es suficiente para todos los modelos usados).

## Metodología de evaluación

- **Bloque bibliográfico:** métricas ya publicadas por cada modelo (PESQ, STOI, SI-SDR, LSD) sobre datasets académicos de referencia (VoiceBank+DEMAND, MUSDB18, DNS Challenge).
- **Bloque empírico:** audio real (grabación del tutor, con reverberación, sin referencia limpia), evaluado con métricas no intrusivas (DNSMOS, NISQA).
- **Comparativas cruzadas:** modelo individual vs. combinado, IA vs. baseline clásico, bloque bibliográfico vs. bloque empírico.



Rodri — Grado en Ingeniería de Telecomunicaciones, ETSIT/UPV.
Tutor: alalbiol.
 
