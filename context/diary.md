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
