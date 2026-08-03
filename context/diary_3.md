# Diario de seguimiento del TFG

Registro cronológico de avances, decisiones y reuniones con el tutor. Cada entrada nueva se añade al final (orden ascendente).

> Nota: las entradas anteriores al 2026-07-13 tienen fechas aproximadas/inventadas para poder llevar un diario con formato de fecha desde el inicio del proyecto; no se registró la fecha exacta en su momento.

Formato sugerido por entrada:

```
## AAAA-MM-DD

**Hecho:**
- ...

**Problemas / bloqueos:**
- ...

**Decisiones tomadas:**
- ...

**Próximos pasos:**
- ...
```

---

## 2026-07-01

**Hecho:**
- Definido el alcance del TFG: 5 categorías de degradación de audio (denoising, dereverberation, super-resolución/BWE, de-clipping, separación de fuentes) + 1 modelo combinado multi-tarea transversal.
- Seleccionados los modelos preentrenados: DeepFilterNet2/3 (denoising), MP-SENet (dereverb), AudioSR (super-resolución), VoiceFixer v2 (de-clipping), HTDemucs v4 (separación de fuentes), ClearVoice/MossFormer2 (combinado).
- Definida la metodología mixta: bloque bibliográfico (métricas publicadas: PESQ, STOI, SI-SDR, LSD) + bloque empírico (audio real del tutor, sin referencia limpia, métricas no intrusivas DNSMOS/NISQA).

**Problemas / bloqueos:**
- Ninguno.

**Decisiones tomadas:**
- No se entrena ningún modelo; solo inferencia con preentrenados de Hugging Face, más un baseline clásico no-IA por categoría.
- Comparativas cruzadas a realizar: individual vs. combinado, IA vs. no-IA, bibliográfico vs. empírico.

**Próximos pasos:**
- Montar el entorno base en Google Colab (Fase 0).

---

## 2026-07-02

**Hecho:**
- Empezado el notebook `00_Setup_Base.ipynb`: montaje de Drive, estructura de carpetas del proyecto, configuración de `HF_HOME`.
- Primeras utilidades de audio (carga, guardado, resampleo).

**Problemas / bloqueos:**
- Ninguno relevante.

**Decisiones tomadas:**
- Carpeta central del proyecto en Drive: `/content/drive/MyDrive/Proyecto_Audio`.
- Utilidades a extraer a `.py` dentro de `utils/` en lugar de mantenerlas solo en el notebook.

**Próximos pasos:**
- Añadir gestión de memoria GPU y métricas.

---

## 2026-07-03

**Hecho:**
- Añadida gestión de memoria GPU en `audio_utils_memoria_GPU.py`.
- Implementadas métricas de referencia (PESQ, STOI, SI-SDR, LSD) y no intrusiva (DNSMOS).
- Implementados los baselines clásicos no-IA por categoría.
- Utils extraídos definitivamente a `.py` en `utils/`.
- **Fase 0 completada.**

**Problemas / bloqueos:**
- Ninguno.

**Decisiones tomadas:**
- Patrón fijo de notebook por modelo a seguir en la Fase 2: instalar dependencias propias → importar utils comunes → inferencia sobre audio de prueba → guardar en `outputs/` → calcular DNSMOS → correr baseline clásico equivalente → `liberar_memoria_gpu` al final.

**Próximos pasos:**
- Empezar `01_DeepFilterNet_3.ipynb` (denoising), primer notebook de la Fase 2.

---

## 2026-07-04

**Hecho:**
- Empezado `01_DeepFilterNet_3.ipynb`.
- Primeros intentos de instalación de `deepfilternet`.

**Problemas / bloqueos:**
- Fallo silencioso de compilación de Rust al instalar con el flag `--no-build-isolation`.
- Conflicto de versiones de numpy (1.26.x en disco vs. 2.0.2 cacheado por Colab).

**Decisiones tomadas:**
- Eliminar el flag `--no-build-isolation`.
- No fijar numpy manualmente antes de instalar `deepfilternet`: la propia librería resuelve su restricción de la serie 1.26.x.

**Próximos pasos:**
- Confirmar que el reinicio de kernel tras la instalación resuelve el conflicto de numpy.

---

## 2026-07-05

**Hecho:**
- Confirmado que el reinicio de kernel tras la instalación es obligatorio y resuelve el conflicto de numpy.
- Añadidos shims de compatibilidad para `torchaudio.backend.common.AudioMetaData` y `torchaudio.info()` (eliminados en `torchaudio ≥2.9`).
- Corregidos imports faltantes en `audio_utils_baselines_clasicos.py` y `audio_utils_metricas_ref.py`.
- Parcheado `calcular_dnsmos` para remuestrear a 16 kHz y convertir estéreo a mono automáticamente (el audio del tutor es estéreo nativo, `(2, 4224000)`).
- Reescrito `liberar_memoria_gpu` usando `inspect.currentframe().f_back` para liberar de verdad la memoria GPU.
- **`01_DeepFilterNet_3.ipynb` depurado y finalizado.**

**Problemas / bloqueos:**
- Resueltos todos los anteriores.

**Decisiones tomadas:**
- No descargar y volver a subir el notebook a mitad de sesión (crea un runtime nuevo y se pierden los paquetes instalados); pegar el código corregido directamente en el notebook ya abierto.

**Próximos pasos:**
- Redactar el apartado de Objetivos de la memoria.
- Empezar el notebook de separación de fuentes (HTDemucs v4).

---

## 2026-07-06

**Hecho:**
- Redactado y finalizado el apartado de **Objetivos** de la memoria: 8 objetivos numerados (revisión del estado del arte, análisis teórico/matemático, selección de modelos e integración de inferencia, definición de métricas de los bloques bibliográfico y empírico, dos análisis comparativos —IA vs. baseline y especializado vs. multi-tarea—, conclusiones y trabajo futuro, y desarrollo de la app Gradio).

**Problemas / bloqueos:**
- Ninguno.

**Decisiones tomadas:**
- Ninguna adicional.

**Próximos pasos:**
- Empezar `02_HTDemucs.ipynb` (separación de fuentes).

---

## 2026-07-07

**Hecho:**
- Empezado `02_HTDemucs.ipynb`.
- Detectado que el paquete `demucs` original de PyPI está abandonado y exige `torch <2.2`.

**Problemas / bloqueos:**
- Incompatibilidad de `demucs` con la versión de torch de Colab.

**Decisiones tomadas:**
- Usar el paquete `demucs-infer` en su lugar.

**Próximos pasos:**
- Completar la inferencia y el baseline clásico de separación de fuentes.

---

## 2026-07-08

**Hecho:**
- **`02_HTDemucs.ipynb` completado** con `demucs-infer`: inferencia sobre audio de prueba, cálculo de DNSMOS, baseline clásico equivalente.

**Problemas / bloqueos:**
- Ninguno relevante.

**Decisiones tomadas:**
- Ninguna adicional.

**Próximos pasos:**
- Empezar el notebook de VoiceFixer v2 (de-clipping).

---

## 2026-07-09

**Hecho:**
- **Notebook de VoiceFixer v2 (de-clipping) completado**: inferencia, DNSMOS, baseline clásico.

**Problemas / bloqueos:**
- Ninguno destacable.

**Decisiones tomadas:**
- Ninguna adicional.

**Próximos pasos:**
- Empezar el notebook de MP-SENet (dereverberation).

---

## 2026-07-10

**Hecho:**
- **Notebook de MP-SENet (dereverberation) completado**: inferencia, DNSMOS, baseline clásico.

**Problemas / bloqueos:**
- Ninguno destacable.

**Decisiones tomadas:**
- Aprovechar que la versión extendida de MP-SENet también soporta BWE y de-clipping como argumento de justificación de su selección por versatilidad (para la memoria).

**Próximos pasos:**
- Empezar `05_AudioSR.ipynb` (super-resolución).

---

## 2026-07-11

**Hecho:**
- Empezado `05_AudioSR.ipynb`.

**Problemas / bloqueos:**
- `audiosr==0.0.7` declara `numpy<=1.23.5`, que no compila en Python 3.12 (Colab actual).
- El modelo solo soporta clips de hasta ~5.12 s (limitación arquitectónica); falla con CUDA OOM si se le pasa audio más largo sin trocear.

**Decisiones tomadas:**
- Instalar con `pip install audiosr==0.0.7 --no-deps` y añadir a mano las dependencias transitorias necesarias (`unidecode`, `phonemizer`, `ftfy`, `torchlibrosa`).
- Implementar chunking en segmentos de ~5 s con solape de 0.1 s y crossfade lineal al recomponer.

**Próximos pasos:**
- Terminar de depurar AudioSR (bug de chunks en silencio).

---

## 2026-07-12

**Hecho:**
- Detectado y resuelto un bug: si un chunk está en silencio o casi silencio, la estimación interna de roll-off espectral de AudioSR degenera y `scipy.iirfilter` lanza `"Wn must be 0 < Wn < 1"`.
- Solución: detectar chunks con RMS por debajo de un umbral y remuestrearlos de forma clásica (`librosa.resample`) en vez de pasarlos por el modelo, con un `try/except` adicional como fallback general.
- Confirmado que AudioSR siempre devuelve la salida en mono y a 48 kHz fijo, independientemente del sample rate/canales de entrada → forzar mono desde la carga inicial del audio.
- **`05_AudioSR.ipynb` completado.**

**Problemas / bloqueos:**
- Resueltos todos los anteriores.

**Decisiones tomadas:**
- Reconocer en las conclusiones que AudioSR ya no es estrictamente SOTA (modelos más nuevos como Vocos-BWE lo superan), pero sigue siendo el baseline estándar de referencia; se documenta como limitación, no como motivo de descarte.
- Patrón general de depuración en utils compartidos: comprobar con `inspect.signature()` o `dir()` los nombres y argumentos reales de las funciones (p. ej. `calcular_dnsmos(audio, sr)` recibe array ya cargado, no una ruta; el baseline de BWE se llama `baseline_bwe_interpolacion_spline(audio, sr_origen, sr_destino)`) en vez de asumirlos.

**Próximos pasos:**
- Empezar el notebook 6: ClearVoice/MossFormer2 (modelo combinado multi-tarea).

---

## 2026-07-13

**Hecho:**
- Creado el repositorio de GitHub del TFG y compartido con el tutor (usuario `alalbiol`).
- Sincronizado el repositorio con la carpeta `Proyecto_Audio` en Drive.
- Creada la carpeta `context/` con `objectives.md`, `TFG_plan.md` y este `diary.md`.
- Reorganizado el diario con fechas por entrada (1-13 de julio) para reflejar el avance de forma cronológica.
- Creado `milestones.md` en `context/` como archivo de hitos de alto nivel, separado del diario.

**Problemas / bloqueos:**
- Ninguno.

**Decisiones tomadas:**
- Uso de `context/` como carpeta central de seguimiento, con archivos `.md` para objetivos, plan y diario.

**Próximos pasos:**
- Empezar notebook `06_ClearVoice_MossFormer2.ipynb` (modelo combinado multi-tarea, usando ClearVoice/MossFormer2).
- Definir con el tutor las fechas objetivo de los milestones en `milestones.md`.

---

## 2026-07-14

**Hecho:**
- **`06_ClearVoice_MossFormer2.ipynb` completado.** Con esto se da por **cerrada la Fase 2** (los 6 notebooks de modelo terminados).
- Inferencia sobre audio real de prueba (voz de niña en catalán, obra de teatro infantil, mucho eco, micrófono lejano) con el pipeline combinado (denoising + dereverb + SR).
- Calculado DNSMOS del resultado combinado y comparado frente al resultado de SE por separado.

**Problemas / bloqueos:**
- Ninguno técnico. Hallazgo a documentar (no es un bloqueo, es un resultado): el pipeline combinado SE+SR dio DNSMOS ligeramente **peor** que solo-SE en las 4 submétricas (ovrl/sig/bak/p808_mos), coincidiendo en dirección con la percepción auditiva ("se escucha considerablemente peor" con SR encima) pero con una magnitud numérica mucho menor de lo que sugiere la percepción subjetiva.

**Decisiones tomadas:**
- **Cambio de patrón para este notebook:** a diferencia de los notebooks 1-5, el notebook 6 (modelo combinado) **no** se compara contra un baseline clásico no-IA propio, sino contra los modelos individuales correspondientes (p. ej. solo-SE). No tiene sentido un "baseline clásico multi-tarea", así que la comparativa individual-vs-combinado sustituye aquí a la comparativa IA-vs-no-IA. **Importante dejarlo explícito en la memoria** (apartado de Metodología o de Desarrollo y resultados) para que no parezca una inconsistencia sino una decisión metodológica justificada.
- El hallazgo de DNSMOS peor-pero-poco se documentará como ejemplo de la discrepancia objetivo/subjetivo, con dos hipótesis a desarrollar: (1) desajuste de dominio de DNSMOS (entrenado sobre DNS Challenge, voz adulta y mayormente en inglés) frente a un caso muy fuera de distribución; (2) posible alucinación/artefactos de alta frecuencia de los modelos SR generativos con inputs muy degradados, que afectan mucho a la percepción humana sin mover tanto las métricas no intrusivas.

**Próximos pasos:**
- Fase 3: recopilar las métricas bibliográficas publicadas de cada modelo (PESQ, STOI, SI-SDR, LSD).
- Empezar a redactar Metodología y Desarrollo y resultados de la memoria, incorporando el cambio de patrón del notebook 6 y el hallazgo del DNSMOS.

---

## 2026-07-15

**Hecho:**
- Subidos `diary.md` y `milestones.md` actualizados a GitHub (commit: cierre de Fase 2 / notebook 6 completado).
- Leído el documento técnico de "Deep Learning for audio signal processing" para contextualizar generalmente. 

**Problemas / bloqueos:**
- Ninguno técnico. Duda pendiente sobre el flujo correcto de clonar/sincronizar el repo entre GitHub y Drive.

**Decisiones tomadas:**
- Ninguna adicional.

**Próximos pasos:**
- Repasar bien el flujo de clonado del repo (GitHub ↔ Drive/Colab).
- Empezar Fase 3: recopilación de métricas bibliográficas.

# 16/07/2026 — Estudio de métricas: STOI y PESQ

## Objetivo de la sesión
Empezar el repaso de las métricas usadas en el TFG (bloque bibliográfico:
PESQ, STOI, SI-SDR, LSD; bloque empírico: DNSMOS, NISQA), antes de redactar
la sección de Metodología.

## Trabajo realizado
- Lectura completa del paper original de **STOI** (Taal et al., 2011). Fijado
  que la implementación usada en el proyecto (`pystoi`, `extended=False`)
  corresponde a esta cita, no a Jensen & Taal 2016 (ESTOI). Anotado que el
  propio paper de STOI reconoce que no fue diseñado para condiciones de
  reverberación — relevante para el bloque empírico (audio del tutor, con
  reverb fuerte).
- Lectura completa del paper original de **PESQ**. Anotado que el propio
  paper reconoce poca correlación con el MOS subjetivo en escenarios de
  clipping/silencio — directamente relevante para la categoría de de-clipping.

## Pendiente
Continuar con SI-SDR y LSD.

---

# 17/07/2026 — Estudio de métricas: SI-SDR y LSD

## Objetivo de la sesión
Continuar el repaso de métricas del bloque bibliográfico.

## Trabajo realizado
- **SI-SDR**: repasado el fundamento matemático completo — α como escalar
  (no vector), ganancia óptima obtenida por proyección ortogonal de la señal
  estimada sobre la referencia, con el error geométricamente perpendicular
  a la referencia (demostrado vía producto escalar = 0), lo que da la
  invarianza a escala.
- **LSD**: repasado el cálculo (RMSE de doble promedio sobre valores de
  energía log-espectral, en log₁₀ sin factor de 10, no estrictamente en dB).
  Anotadas las precauciones: normalizar energía antes de calcular LSD, y el
  problema de comparabilidad entre papers cuando usan convenciones distintas
  (log₁₀ vs. 10·log₁₀). Fijada la cita de AERO (Mandel, Tal & Adi, 2022,
  arXiv:2211.12232) en vez del paper de Gray & Markel, inaccesible.

## Pendiente
Continuar con DNSMOS y NISQA (bloque empírico).

---

# 18/07/2026 — Estudio de métricas: DNSMOS y NISQA

## Objetivo de la sesión
Cerrar el repaso de métricas con las dos no-intrusivas del bloque empírico.

## Trabajo realizado
- **DNSMOS**: leída la parte descriptiva del paper (arXiv:2110.01763) a partir
  de un resumen, para entender el funcionamiento general (estimación de MOS
  sin referencia limpia, submétricas SIG/BAK/OVRL).
- **NISQA**: mismo enfoque — resumen + lectura de la parte descriptiva del
  paper (arXiv:2104.09494), como comparativa con DNSMOS al ser de la misma
  familia (predictores neuronales de MOS no-intrusivos).

## Pendiente
Profundizar en DNSMOS y NISQA antes de redactar Metodología (según lo acordado,
tratamiento en profundidad para DNSMOS y comparativo para NISQA). Con las 6
métricas ya repasadas al menos una vez, el siguiente paso natural es empezar
a redactar la sección de Metodología.

---

# 28/07/2026 — Lectura del paper de DeepFilterNet

## Objetivo de la sesión
Revisar en profundidad el modelo de denoising seleccionado, antes de conectar
su inferencia en la app.

## Trabajo realizado
- Lectura del paper de DeepFilterNet (arquitectura DeepFilterNet2/3), apoyada
  con resúmenes para facilitar la digestión de las partes más densas.
- Sesión realizada desde otro dispositivo, usando Gemini como apoyo.

## Pendiente
Continuar con el paper de HTDemucs (siguiente modelo en el orden de la Fase 2).

---

# 29/07/2026 — Lectura del paper de HTDemucs

## Objetivo de la sesión
Revisar en profundidad el modelo de separación de fuentes seleccionado.

## Trabajo realizado
- Lectura del paper de HTDemucs v4, apoyada con resúmenes para facilitar la
  digestión de las partes más densas.
- Sesión realizada desde otro dispositivo, usando Gemini como apoyo.

## Pendiente
Con DeepFilterNet y HTDemucs ya revisados a nivel de paper, continuar con el
resto de modelos (MP-SENet, AudioSR, VoiceFixer, ClearVoice/MossFormer2) y
retomar el desarrollo de la app (Fase 4).

# 31/07/2026 — Primera versión funcional de la app (Gradio, categoría Denoising)

## Objetivo de la sesión
Cambio de orden en el plan de trabajo: adelantar el desarrollo de la aplicación
(Fase 4) antes de empezar a redactar la memoria, para tener ya una demo funcional
y contenido real (comportamiento del pipeline, casos límite con audio real) de
cara a la sección de diseño de la app y a los resultados.

## Diseño de la interfaz
- Interfaz Gradio en dos niveles: Nivel 1 = categoría de degradación (denoising,
  dereverberation, super-resolución, de-clipping, separación de fuentes), Nivel 2 =
  modelo dentro de esa categoría (individual vs. combinado MossFormer2, cuando aplique).
- Gestión de memoria: un único modelo cargado en GPU a la vez, liberando el anterior
  antes de cargar el siguiente (mismo patrón que `liberar_memoria_gpu` en las notebooks
  de Colab, generalizado en un `model_manager.py` para reutilizar con los 6 modelos).

## Implementación
- Primer wrapper de inferencia conectado: **DeepFilterNet3** (denoising), reutilizando
  la lógica de la notebook `01_DeepFilterNet_3.ipynb` (shims de compatibilidad de
  torchaudio, `init_df()`, `enhance()`).
- Baseline clásico (spectral gating) integrado en el mismo flujo, calculado siempre
  junto al resultado del modelo IA para la comparativa IA vs. no-IA.
- Métricas DNSMOS calculadas sobre las tres versiones (original, IA, baseline).

## Bugs encontrados y resueltos
1. **Colisión de nombres de paquete**: las carpetas `utils/` y `models/` colisionaban
   con paquetes genéricos ya presentes en el entorno de Colab (probablemente
   dependencias de `gradio`). Solución: renombradas a `tfg_audio_utils/` y
   `tfg_models/`.
2. **Orden de aplicación de los shims de torchaudio**: el parche de compatibilidad
   (`AudioMetaData`, `torchaudio.info()`, necesario por la eliminación de estas
   APIs en torchaudio >=2.9) se aplicaba dentro de la función de carga del modelo,
   pero un import de `df.enhance` en otra función se ejecutaba antes, rompiendo la
   importación. Solución: aplicar el shim a nivel de módulo, garantizando que se
   ejecute antes de cualquier import de `df`/`torchaudio`.
3. **Guardado incorrecto del audio procesado**: se usaba el `guardar_audio` genérico
   (basado en `soundfile`) para la salida del modelo IA, cuando la notebook original
   usaba `save_audio` (de `df.enhance`), pensado específicamente para el tensor que
   devuelve `enhance()`. Causaba `LibsndfileError: Format not recognised`.

## Hallazgo relevante para el bloque empírico (memoria)
Al probar DeepFilterNet3 con el audio real del tutor (reverb fuerte, mic distante),
el modelo generó picos de amplitud muy por encima de 1.0 (hasta el punto de que,
al guardarse sin normalizar, se producía *wraparound* al convertir a int16 — crujido
de saturación audible, con volumen pegado casi al máximo en la práctica totalidad
del audio). Solucionado normalizando el pico antes de guardar (`normalizar_pico`,
ya existente en los utils).

Este comportamiento es coherente con el desajuste de dominio ya identificado en la
notebook 06 (DeepFilterNet3 entrenado sobre DNS Challenge: voz adulta, mayormente
inglés, condiciones de grabación controladas): con reverb real fuerte, el modelo
puede sobre-amplificar o "alucinar" corrección de forma inestable. Las métricas
DNSMOS, sin embargo, no reflejan bien este problema perceptual:

| Versión              | OVRL  | SIG   | BAK   |
|-----------------------|-------|-------|-------|
| Original (degradado) | 1.385 | 1.578 | 1.790 |
| DeepFilterNet3 (IA)   | 2.090 | 2.309 | 3.747 |
| Baseline (no-IA)      | 1.400 | 1.578 | 1.843 |

DNSMOS puntúa la salida de DeepFilterNet muy por encima del original y del baseline
pese a la saturación perceptual evidente — un segundo caso (en sentido inverso al de
la notebook 06) de discrepancia entre métrica no-intrusiva y percepción subjetiva,
que refuerza el argumento central de la metodología dual (bibliográfico + empírico)
del TFG.

## Pendiente / próximos pasos
- Revisar si sigue habiendo un recorte de audio al inicio de la salida IA (síntoma
  reportado, pendiente de confirmar si es real o percepción).
- Conectar el siguiente modelo (HTDemucs v4, separación de fuentes).
- Documentar este hallazgo con más detalle (qué se oye exactamente, si varía según
  la intensidad de la reverb en cada tramo) para la sección de resultados.

  
## 02/08/2026 — App funcional: depuración a fondo del pipeline de denoising
Objetivo de la sesión

Terminar de dejar 100% funcional la categoría de denoising en la app (DeepFilterNet3), tras los primeros bugs detectados en sesiones anteriores. Sesión larga, con varios problemas encadenados de entorno y de código.

# Reorganización de notebooks
Se separó definitivamente la lógica de la app de la notebook de validación del modelo: 01_DeepFilterNet_3_6.ipynb vuelve a contener solo la validación individual de DeepFilterNet3 (bloque bibliográfico/empírico), sin celdas de Gradio.
Nueva notebook dedicada app_gradio.ipynb: instala dependencias e inicia la app de Gradio. Es la que crecerá con la instalación de cada modelo nuevo que se conecte.
# Bugs de entorno (Colab) resueltos
Versión de Gradio inestable: al no fijar versión, cada sesión instalaba una distinta, causando estilos rotos y desconexiones del túnel share=True.
Bug real de Gradio 4.44.1 y 5.9.1: un TypeError: argument of type 'bool' is not iterable en gradio_client/utils.py (función get_type), reproducible con ambas versiones pese a tener las parejas gradio/gradio_client correctamente emparejadas. Se comprobó que la versión 6.20.0 no sufre este bug. Fijada gradio==6.20.0 en requirements.txt y en la celda de instalación de la notebook de la app.
Conflicto de huggingface_hub: versión moderna de Colab incompatible con Gradio 4.44.1 (eliminó HfFolder). Dejó de ser relevante al pasar a Gradio 6.20.0.
Numpy roto por instalar Gradio: instalar gradio==6.20.0 arrastra una versión de pandas que exige numpy>=2, rompiendo la extensión nativa de deepfilternet (compilada contra numpy 1.26.4). Solución: reinstalar numpy==1.26.4 justo después de instalar gradio, seguido de un reinicio de runtime.
show_api no soportado en Gradio 6.20.0: parámetro específico de versiones anteriores, eliminado del launch() en la 6.x. Quitado de las llamadas.
# Bug de código encontrado y corregido (importante)
Causa raíz de soundfile.LibsndfileError: Format not recognised: DeepFilterNet3 devuelve el audio como tensor con forma (canales, muestras) para audio estéreo. soundfile.write() (usado en guardar_audio) espera el orden contrario, (muestras, canales). Al no transponer, sf.write no reconocía el formato. Confirmado con un print de diagnóstico (shape=(2, 4224000)) antes de arreglarlo a ciegas.
Explica también por qué usar save_audio (de df.enhance, basado en torchaudio, que sí espera (canales, muestras)) "funcionaba" en versiones anteriores del código: cada función de guardado esperaba un orden de ejes distinto.
Fix aplicado en tfg_models/denoising.py: transponer el array (.T) antes de guardar únicamente cuando tiene 2 dimensiones (estéreo), manteniendo guardar_audio (con control total sobre la normalización de pico, a diferencia de save_audio, que parecía re-normalizar internamente el pico al guardar).
# Estado final

Categoría de Denoising (DeepFilterNet3) funcionando de extremo a extremo en la app: selección de categoría/modelo (nivel 1/nivel 2), inferencia, comparación con baseline clásico, tabla de métricas DNSMOS, y guardado correcto del audio de salida (mono y estéreo) sin distorsión por desbordamiento ni error de formato.

## 03-08-2026

**Fase 4 — App Gradio: MP-SENet (dereverberation) y VoiceFixer v2 (de-clipping) conectados**

- Se conecta MP-SENet a la app siguiendo el mismo patrón que `denoising.py`
  (paquete `MPSENet`, checkpoint `JacobLinCool/MP-SENet-VB`, vía
  `model_manager.obtener_modelo("dereverberation", "mpsenet", ...)`). Notebook
  `app_gradio.ipynb` actualizada con la celda de instalación (`pip install
  MPSENet`, sin Rust ni reinicio de kernel).
- **Hallazgo importante (para la memoria):** con el audio real del tutor
  (reverb fuerte, mic distante), la salida de MP-SENet suena a ruido blanco
  casi constante, con solo 2 fragmentos donde se entiende algo de voz — pero
  DNSMOS puntúa la IA por encima del baseline clásico. Causa raíz: el paquete
  pip `MPSENet` solo expone los checkpoints `g_best_dns`/`g_best_vb` del
  MP-SENet base, que son de **denoising** (DNS-Challenge / VoiceBank+DEMAND),
  no de dereverberation. La "versión larga" del repo original (yxlu-0102) sí
  cubre dereverb+BWE, pero no está empaquetada en el pip usado — no se
  persigue esa vía por tiempo, se documenta como limitación y como el caso
  más severo de domain mismatch de los tres encontrados hasta ahora (junto a
  DeepFilterNet3 y MossFormer2).
- Se detecta y corrige un bug de configuración: `login(token=...)` de HF no
  persiste entre reinicios de kernel de Colab — había que moverlo a la celda
  posterior al *último* reinicio (justo antes de lanzar la app), no a la
  primera celda tras montar Drive.
- Se conecta VoiceFixer v2 a la app (`declipping.py`, nuevo). A diferencia de
  DeepFilterNet3/MP-SENet, la API de VoiceFixer es archivo-a-archivo
  (`restore(input=..., output=...)`, sin devolver array en memoria) — el
  wrapper escribe a un temporal, relee con `cargar_audio()`, normaliza y
  calcula DNSMOS igual que el resto. Salida fija a 44100 Hz. Instalación con
  `--no-deps` + dependencias sueltas (`torchlibrosa`, `progressbar`,
  `GitPython`, `pyyaml`) para evitar que el `setup.py` de `voicefixer`
  reinstale versiones antiguas de `librosa`/`matplotlib` (mismo tipo de
  precaución que con AudioSR en la notebook 05).
- Refactor de `app.py`: la selección de función de inferencia y de baseline
  clásico pasa de `if/elif` hardcodeado a dos diccionarios
  (`FUNCIONES_INFERENCIA`, `BASELINES_CLASICOS`), para que conectar AudioSR y
  HTDemucs sea solo añadir una entrada. Nota: `baseline_declipping_interpolacion_cubica`
  tiene firma distinta (`audio, umbral=0.99`, sin `sr`) — se envuelve en un
  lambda para mantener la interfaz común `baseline(audio, sr)`.
- Pendiente de probar por el usuario: VoiceFixer v2 con audio real
  (de-clipping) — sesión de pruebas queda para más tarde.

**Nota sobre selección de audio de prueba por categoría (pendiente de aplicar
en próximas sesiones):** el audio real del tutor (reverb+eco) solo es
representativo para Dereverberation y para el combinado MossFormer2. Para el
resto hace falta audio específico de cada degradación:
- *Denoising*: ruido aditivo real (no reverb).
- *BWE*: audio con ancho de banda limitado (nota de voz comprimida, llamada,
  o generarlo synthetically downsampleando una grabación limpia — mismo
  método ya usado en la notebook 05).
- *De-clipping*: audio con saturación real o generado sintéticamente
  (amplificar por encima de ±1.0 y hard-clip).
- *Separación de fuentes (HTDemucs)*: necesita música con voz + instrumentos,
  no voz hablada sola — cualquier canción con acompañamiento vale.

**Siguiente en cola:** conectar HTDemucs v4 (separación de fuentes) en la app.
