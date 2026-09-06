# TFG — Restauración interactiva de señales de audio degradadas mediante aprendizaje profundo

TFG del Grado en Ingeniería de Telecomunicaciones (ETSIT, Universitat Politècnica de València).

**Título completo:** *Análisis y aplicación de modelos de aprendizaje profundo para la restauración interactiva de señales de audio degradadas.*

**Autor:** Rodri — Tutor: alalbiol

## Descripción

El proyecto aborda cinco categorías de degradación de audio — eliminación de ruido (*denoising*), eliminación de reverberación (*dereverberation*), super-resolución o ampliación de ancho de banda (*BWE*), corrección de recorte (*de-clipping*) y separación de fuentes — mediante la aplicación de modelos de aprendizaje profundo **preentrenados** de Hugging Face, sin entrenamiento propio.

Para cada categoría se emplea un modelo especializado, más un modelo combinado multi-tarea y un baseline clásico no basado en IA, comparados mediante métricas bibliográficas (PESQ, STOI, SI-SDR, LSD) y métricas empíricas no intrusivas (DNSMOS, NISQA) sobre audio real sin referencia limpia. Como complemento, se incluye un bloque de audios sintéticos degradados de forma controlada (barrido de severidad por categoría), que permite emplear también métricas intrusivas allí donde el audio real no dispone de referencia limpia.

El resultado final se integra en una aplicación interactiva con Gradio (`gradio_app/app.py`), ejecutada sobre Google Colab (GPU) mediante el notebook `gradio_app/app_gradio_venv311.ipynb`, con un túnel `ngrok` para acceder a ella desde fuera de la propia sesión de Colab. El despliegue como Hugging Face Space se planteó inicialmente pero se descartó por motivos de tiempo (ver `context/TFG_plan.md`); queda documentado como posible línea de trabajo futuro.

## Modelos utilizados

| Categoría | Modelo |
|---|---|
| Denoising | DeepFilterNet2/3 |
| Dereverberation | SGMSE+ |
| Super-resolución (BWE) | AudioSR |
| De-clipping | VoiceFixer v2 |
| Separación de fuentes | HTDemucs v4 |
| Combinado multi-tarea | ClearVoice / MossFormer2 |

Cada categoría incluye además un baseline clásico (no basado en IA) como referencia comparativa.

> **Nota sobre dereverberation:** el paquete de PyPI de MP-SENet solo distribuye
> checkpoints de denoising, sin soporte nativo para dereverberation. Por ello, el modelo
> activo de dereverberation en la aplicación final es **SGMSE+** (sí entrenado
> específicamente para esta tarea); MP-SENet se mantiene en el repo (`mpsenet_dereverb.py`)
> y documentado en la memoria como caso de estudio de esta misma limitación (apartado 4.4.2).

## Estructura del repositorio

```
TFG_restauracion_audio/
├── README.md
├── .gitignore
├── GIT_CHEATSHEET.md               # Comandos git de referencia rápida
├── context/                        # Seguimiento del proyecto (para el tutor)
│   ├── objectives.md
│   ├── TFG_plan.md
│   ├── milestones.md
│   └── diary.md
├── gradio_app/                     # Aplicación interactiva
│   ├── app.py                       # Interfaz Gradio (dos niveles: categoría → modelo)
│   ├── app_gradio_venv311.ipynb     # Notebook de instalación y lanzamiento en Colab
│   ├── tfg_models/                  # Wrappers de inferencia por modelo
│   │   ├── model_manager.py          # Un único modelo cargado en GPU a la vez
│   │   ├── denoising.py
│   │   ├── dereverb.py                # SGMSE+ (variante activa)
│   │   ├── mpsenet_dereverb.py        # MP-SENet (caso de estudio, ver 4.4.2)
│   │   ├── bwe.py
│   │   ├── declipping.py
│   │   ├── separacion_fuentes.py
│   │   └── mossformer2.py
│   ├── tfg_audio_utils/             # Utilidades compartidas
│   │   ├── audio_utils_funcionescomunes.py
│   │   ├── audio_utils_baselines_clasicos.py
│   │   ├── audio_utils_metricas_ref.py
│   │   ├── audio_utils_metricas_no_intrusivas.py
│   │   └── audio_utils_memoria_GPU.py
│   └── pesos/                       # Checkpoints descargados (SGMSE+); NO en git, ver .gitignore
├── notebooks/                       # Notebooks de validación individual por modelo (Fase 1)
│   ├── 00_Setup_Base.ipynb
│   ├── 01_DeepFilterNet_3.ipynb
│   ├── 02_HTDemucs.ipynb
│   ├── 03_VoiceFixer.ipynb
│   ├── 04_MPSENet.ipynb
│   ├── 05_AudioSR.ipynb
│   └── 06_ClearVoice_MossFormer2.ipynb
├── generar_audios_prueba.py         # Genera el barrido sintético de audios degradados + manifiesto CSV
└── calcular_metricas.py             # Calcula PESQ/STOI/SI-SDR/LSD/DNSMOS sobre el barrido (reanudable)
```

> Las carpetas `gradio_app/pesos/` (checkpoints descargados), `cache/` (caché de Hugging
> Face) y `outputs/`, así como los audios sintéticos generados por
> `generar_audios_prueba.py`, se generan localmente al ejecutar el notebook/scripts y
> están excluidos del repositorio (ver `.gitignore`) por su tamaño.

## Estado actual del proyecto

Seguimiento detallado, hitos y fechas en [`context/TFG_plan.md`](context/TFG_plan.md) y [`context/milestones.md`](context/milestones.md).

## Cómo ejecutar la aplicación (Google Colab)

La app está pensada para ejecutarse en **Google Colab con GPU** (T4 es suficiente). Toda la
instalación y el lanzamiento están automatizados en un único notebook:
**[`gradio_app/app_gradio_venv311.ipynb`](gradio_app/app_gradio_venv311.ipynb)**. Basta con
clonar el repositorio en Drive, configurar dos secretos (tokens propios, ver más abajo) y
ejecutar el notebook celda a celda de principio a fin.

### 1. Clonar el repositorio en tu Google Drive

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

### 2. Configurar dos secretos de Colab (tokens propios)

Antes de ejecutar el notebook, cada persona que lo use debe añadir **sus propios** tokens
como secretos de Colab (icono de la llave 🔑 en la barra lateral izquierda → *Add new
secret* → activar *Notebook access*):

| Nombre del secreto | Para qué sirve | Cómo conseguirlo |
|---|---|---|
| `TOKEN_TFG_CLEARMOSS` | Descarga de los checkpoints de Hugging Face (MossFormer2, MP-SENet, etc.) | [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) — token de lectura (*read*) |
| `NGROK_AUTH_TOKEN` | Publicar la app fuera de la sesión de Colab mediante un túnel | Cuenta gratuita en [ngrok.com](https://ngrok.com) → *Your Authtoken* en el dashboard |

Sin el token de Hugging Face la app funciona igual, solo que las descargas de checkpoints
son algo más lentas. **Sin el token de ngrok la última celda del notebook fallará**, ya que
es imprescindible para publicar la URL de la app.

> El notebook lee ambos tokens desde los secretos de Colab (`userdata.get(...)`); en ningún
> momento hay que escribir un token directamente en una celda.

### 3. Ejecutar el notebook

Abrir `gradio_app/app_gradio_venv311.ipynb` en Colab (entorno de ejecución con GPU T4) y
ejecutar las celdas en orden. A alto nivel, el notebook:

1. Monta Drive y fija `PROJECT_ROOT` a la ruta donde se clonó el repo (única variable a
   ajustar si se clonó en otra ubicación distinta de `MyDrive/TFG_restauracion_audio`).
2. Crea un entorno virtual aislado de **Python 3.11** (`/content/env311`) e instala PyTorch
   con la misma versión de CUDA que trae el runtime de Colab. Esto es necesario porque Colab
   puede cambiar la versión de Python por defecto de sus máquinas sin previo aviso (ya
   ocurrió de 3.12 a 3.13 durante el desarrollo de este proyecto), y varias dependencias de
   la app fijan `numpy<2.0`, para el que no hay wheels precompiladas fuera de Python ≤3.11.
3. Instala, en un orden concreto y con algunos `--no-deps` puntuales, las dependencias de
   cada modelo (DeepFilterNet3 necesita además el toolchain de Rust, que el propio notebook
   instala). El orden importa: varios paquetes (`audiosr`, `voicefixer`) declaran en su
   propio `setup.py` versiones de `numpy`/`librosa`/`transformers` que solo usan sus demos de
   línea de comandos (no el código que llama esta app), así que los avisos de conflicto de
   pip durante la instalación son esperados — la celda de comprobación final
   (`check_versiones.py`) es la que de verdad confirma si cada modelo importa bien.
4. Lee los dos secretos de Colab (Hugging Face y ngrok).
5. Clona e instala SGMSE+ y descarga su checkpoint (WSJ0-REVERB) a `gradio_app/pesos/`.
6. Lanza la app en segundo plano (`gradio_app/app.py`, vía el venv de Python 3.11) y publica
   una URL pública con `ngrok.connect(...)`.

La URL de ngrok que imprime la última celda es la que hay que abrir para usar la app.

### Requisitos

- Cuenta de Google con acceso a Google Colab y Google Drive.
- Runtime de Colab con GPU (T4 es suficiente).
- Token de Hugging Face y token de ngrok propios, como secretos de Colab (ver punto 2).
- Un archivo de audio de prueba (propio) para subir a la app — no se incluye ninguno en el
  repositorio.

## Metodología de evaluación

- **Bloque bibliográfico:** métricas ya publicadas por cada modelo (PESQ, STOI, SI-SDR, LSD) sobre datasets académicos de referencia (VoiceBank+DEMAND, MUSDB18, DNS Challenge), usados como referencia del estado del arte, no como parte del bloque empírico propio.
- **Bloque empírico (audio real):** audio real (grabación con reverberación, sin referencia limpia), evaluado con métricas no intrusivas (DNSMOS, NISQA) directamente desde la app.
- **Bloque de audios sintéticos:** barrido de severidad controlado por categoría (SNR, T60, frecuencia de corte, % de saturación, SIR) sobre fragmentos limpios de referencia, generado con `generar_audios_prueba.py` y evaluado con `calcular_metricas.py` (PESQ, STOI, SI-SDR, LSD, DNSMOS), al disponer de referencia limpia conocida. No se calculan baselines clásicos no-IA sobre este bloque: la comparativa IA vs. no-IA ya queda cubierta con los resultados sobre el audio real.
- **Comparativas cruzadas:** modelo individual vs. combinado, IA vs. baseline clásico, bloque bibliográfico vs. bloques empíricos (real y sintético).

## Documentación adicional

- [`context/objectives.md`](context/objectives.md) — objetivos del TFG
- [`context/TFG_plan.md`](context/TFG_plan.md) — plan de fases y milestones
- [`context/milestones.md`](context/milestones.md) — estado de alto nivel por fase/hito
- [`context/diary.md`](context/diary.md) — diario de seguimiento
- [`GIT_CHEATSHEET.md`](GIT_CHEATSHEET.md) — comandos git de referencia
