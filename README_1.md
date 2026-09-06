# TFG — Restauración interactiva de señales de audio degradadas mediante aprendizaje profundo

TFG del Grado en Ingeniería de Telecomunicaciones (ETSIT, Universitat Politècnica de València).

**Título completo:** *Análisis y aplicación de modelos de aprendizaje profundo para la restauración interactiva de señales de audio degradadas.*

**Autor:** Rodri — Tutor: alalbiol

## Descripción

El proyecto aborda cinco categorías de degradación de audio — eliminación de ruido (*denoising*), eliminación de reverberación (*dereverberation*), super-resolución o ampliación de ancho de banda (*BWE*), corrección de recorte (*de-clipping*) y separación de fuentes — mediante la aplicación de modelos de aprendizaje profundo **preentrenados** de Hugging Face, sin entrenamiento propio.

Para cada categoría se emplea un modelo especializado, más un modelo combinado multi-tarea y un baseline clásico no basado en IA, comparados mediante métricas bibliográficas (PESQ, STOI, SI-SDR, LSD) y métricas empíricas no intrusivas (DNSMOS, NISQA) sobre audio real sin referencia limpia. Como complemento, se incluye un bloque de audios sintéticos degradados de forma controlada (barrido de severidad por categoría), que permite emplear también métricas intrusivas allí donde el audio real no dispone de referencia limpia.

El desarrollo técnico se realiza en Google Colab (GPU) y el resultado final se integra en una aplicación interactiva con Gradio, ejecutada y probada en el propio Colab (o en un entorno local equivalente) mediante un túnel `ngrok`. El despliegue como Hugging Face Space se planteó inicialmente pero se descartó por motivos de tiempo (ver `context/TFG_plan.md`); queda documentado como posible línea de trabajo futuro.

## Modelos utilizados

| Categoría | Modelo |
|---|---|
| Denoising | DeepFilterNet2/3 |
| Dereverberation | SGMSE+ |
| Super-resolución (BWE) | AudioSR |
| De-clipping | VoiceFixer v2 |
| Separación de fuentes | HTDemucs v4 |
| Combinado multi-tarea | ClearVoice / MossFormer2 |

Cada notebook incluye además un baseline clásico (no basado en IA) como referencia comparativa.

> **Nota sobre dereverberation:** el paquete de PyPI de MP-SENet solo distribuye
> checkpoints de denoising, sin soporte nativo para dereverberation. Por ello, el modelo
> activo de dereverberation en la aplicación final es **SGMSE+** (sí entrenado
> específicamente para esta tarea); MP-SENet se mantiene documentado en la memoria como
> caso de estudio de esta misma limitación.

## Estructura del repositorio

```
TFG_restauracion_audio/
├── README.md
├── .gitignore
├── GIT_CHEATSHEET.md               # Comandos git de referencia rápida
├── context/                        # Seguimiento del proyecto (para el tutor)
│   ├── objectives.md                # Objetivos del TFG
│   ├── TFG_plan.md                  # Plan de fases y milestones
│   ├── milestones.md                # Estado de alto nivel por fase/hito
│   └── diary.md                     # Diario de avances
├── tfg_audio_utils/                # Utilidades compartidas (Python)
│   ├── audio_utils_funcionescomunes.py
│   ├── audio_utils_baselines_clasicos.py
│   ├── audio_utils_metricas_ref.py
│   ├── audio_utils_metricas_no_intrusivas.py
│   └── audio_utils_memoria_GPU.py
├── tfg_models/                     # Wrappers de inferencia por modelo (usados por la app)
│   ├── model_manager.py             # Carga/descarga de un único modelo en GPU a la vez
│   ├── denoising.py
│   ├── dereverberation.py
│   ├── bwe.py
│   ├── declipping.py
│   ├── separacion_fuentes.py
│   └── combinado_mossformer2.py
├── app.py                          # Aplicación Gradio (interfaz de dos niveles)
├── app_gradio.ipynb                # Notebook de instalación y lanzamiento de la app en Colab
├── 00_Setup_Base.ipynb
├── 01_DeepFilterNet_3.ipynb
├── 02_HTDemucs.ipynb
├── 03_VoiceFixer.ipynb
├── 04_MPSENet.ipynb
├── 05_AudioSR.ipynb
├── 06_ClearVoice_MossFormer2.ipynb
├── generar_audios_prueba.py         # Genera el barrido sintético de audios degradados + manifiesto CSV
└── calcular_metricas.py             # Calcula PESQ/STOI/SI-SDR/LSD/DNSMOS sobre el barrido (reanudable)
```

> Las carpetas `outputs/` (audios procesados), `cache/` (pesos de modelos descargados) y
> los audios sintéticos generados por `generar_audios_prueba.py` se generan localmente al
> ejecutar los notebooks/scripts y están excluidos del repositorio (ver `.gitignore`) por
> su tamaño.

## Estado actual del proyecto

Seguimiento detallado, hitos y fechas en [`context/TFG_plan.md`](context/TFG_plan.md) y [`context/milestones.md`](context/milestones.md).

## Cómo ejecutar este proyecto (Google Colab)

Estos notebooks y la aplicación están pensados para ejecutarse en **Google Colab con GPU**. No dependen de una ruta de Drive concreta: basta con clonar el repositorio en cualquier carpeta de tu Google Drive y ajustar una única variable al principio de cada notebook.

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

A partir de aquí, el resto del notebook usa rutas relativas a `PROJECT_ROOT` (por ejemplo, `f"{PROJECT_ROOT}/tfg_audio_utils"` o `f"{PROJECT_ROOT}/outputs"`), por lo que funciona igual sin importar en qué cuenta de Drive o carpeta esté clonado.

### 3. Entorno virtual de Python 3.11 (`env311`)

Colab puede cambiar en cualquier momento la versión de Python por defecto de sus máquinas (ya ocurrió de 3.12 a 3.13 durante el desarrollo de este proyecto), rompiendo la instalación de dependencias como `deepfilternet`, `audiosr`, `clearvoice` o `sgmse`, todas fijadas a `numpy<2.0` sin wheels precompiladas fuera de Python ≤3.11. Para evitarlo, la app se instala y ejecuta dentro de un entorno virtual propio de Python 3.11, creado en la propia máquina virtual de Colab:

```python
!apt-get install -y python3.11 python3.11-venv
!python3.11 -m venv /content/env311
!/content/env311/bin/pip install -r {PROJECT_ROOT}/requirements.txt
```

La celda de instalación de `app_gradio.ipynb` ya incluye este paso; no requiere intervención manual salvo la primera vez.

### 4. Lanzar la aplicación (Gradio + túnel `ngrok`)

La app (`app.py`) expone una interfaz de dos niveles (categoría de degradación → modelo o baseline) para cargar audio, aplicar el modelo seleccionado y visualizar audio antes/después, espectrogramas comparativos y métricas.

Para probarla fuera de la propia sesión de Colab se usa un túnel `ngrok` (más estable en este proyecto que el túnel por defecto de Gradio, `share=True`, que mostró desconexiones frecuentes en algunas sesiones). Requiere una cuenta gratuita de ngrok y su token de autenticación:

```python
from pyngrok import ngrok
ngrok.set_auth_token("TU_TOKEN_DE_NGROK")
```

Ejecutar `app_gradio.ipynb` desde el principio (o, dentro de él, solo la celda de lanzamiento si ya se instalaron las dependencias) lanza la app y publica la URL del túnel. Se recomienda no relanzar `launch()` sin cerrar antes la instancia previa (`demo.close()`), para evitar procesos de Gradio acumulados en la misma sesión de Colab.

> El despliegue como Hugging Face Space no se ha realizado en este proyecto (descartado
> por tiempo); la vía soportada para ejecutar la app es Colab (o un entorno local
> equivalente con Python 3.11 y las mismas dependencias).

### Requisitos

- Cuenta de Google con acceso a Google Colab y Google Drive.
- Runtime de Colab con GPU (T4 es suficiente).
- Cuenta gratuita de ngrok (token de autenticación) para exponer la app fuera de la sesión de Colab.
- Para el bloque empírico sobre audio real: audio de prueba con degradación real (no incluido en el repo por privacidad/tamaño; contactar con el autor si es necesario).
- Para el bloque de audios sintéticos: `generar_audios_prueba.py` y `calcular_metricas.py` pueden ejecutarse también fuera de Colab (por ejemplo, en un entorno local de Windows/VS Code con un entorno virtual propio de Python), siempre que se disponga de las mismas dependencias de audio; `calcular_metricas.py` sí necesita GPU para la inferencia de los modelos.

## Metodología de evaluación

- **Bloque bibliográfico:** métricas ya publicadas por cada modelo (PESQ, STOI, SI-SDR, LSD) sobre datasets académicos de referencia (VoiceBank+DEMAND, MUSDB18, DNS Challenge), usados como referencia del estado del arte, no como parte del bloque empírico propio.
- **Bloque empírico (audio real):** audio real (grabación con reverberación, sin referencia limpia), evaluado con métricas no intrusivas (DNSMOS, NISQA).
- **Bloque de audios sintéticos:** barrido de severidad controlado por categoría (SNR, T60, frecuencia de corte, % de saturación, SIR) sobre fragmentos limpios de referencia, evaluado con métricas intrusivas (PESQ, STOI, SI-SDR, LSD) y DNSMOS, gracias a disponer de referencia limpia conocida. No se calculan baselines clásicos no-IA sobre este bloque: la comparativa IA vs. no-IA ya queda cubierta con los resultados sobre el audio real.
- **Comparativas cruzadas:** modelo individual vs. combinado, IA vs. baseline clásico, bloque bibliográfico vs. bloques empíricos (real y sintético).

## Documentación adicional

- [`context/objectives.md`](context/objectives.md) — objetivos del TFG
- [`context/TFG_plan.md`](context/TFG_plan.md) — plan de fases y milestones
- [`context/milestones.md`](context/milestones.md) — estado de alto nivel por fase/hito
- [`context/diary.md`](context/diary.md) — diario de seguimiento
- [`GIT_CHEATSHEET.md`](GIT_CHEATSHEET.md) — comandos git de referencia
