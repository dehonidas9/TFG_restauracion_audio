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

---

## 2026-07-16 — Estudio de métricas: STOI y PESQ

**Hecho:**
- Lectura completa del paper original de STOI (Taal et al., 2011). Fijado que la implementación usada en el proyecto (`pystoi`, `extended=False`) corresponde a esta cita, no a Jensen & Taal 2016 (ESTOI). Anotado que el propio paper de STOI reconoce que no fue diseñado para condiciones de reverberación — relevante para el bloque empírico (audio del tutor, con reverb fuerte).
- Lectura completa del paper original de PESQ. Anotado que el propio paper reconoce poca correlación con el MOS subjetivo en escenarios de clipping/silencio — directamente relevante para la categoría de de-clipping.

**Próximos pasos:**
- Continuar con SI-SDR y LSD.

---

## 2026-07-17 — Estudio de métricas: SI-SDR y LSD

**Hecho:**
- SI-SDR: repasado el fundamento matemático completo — α como escalar (no vector), ganancia óptima obtenida por proyección ortogonal de la señal estimada sobre la referencia, con el error geométricamente perpendicular a la referencia (demostrado vía producto escalar = 0), lo que da la invarianza a escala.
- LSD: repasado el cálculo (RMSE de doble promedio sobre valores de energía log-espectral, en log₁₀ sin factor de 10, no estrictamente en dB). Anotadas las precauciones: normalizar energía antes de calcular LSD, y el problema de comparabilidad entre papers cuando usan convenciones distintas (log₁₀ vs. 10·log₁₀). Fijada la cita de AERO (Mandel, Tal & Adi, 2022, arXiv:2211.12232) en vez del paper de Gray & Markel, inaccesible.

**Próximos pasos:**
- Continuar con DNSMOS y NISQA (bloque empírico).

---

## 2026-07-18 — Estudio de métricas: DNSMOS y NISQA

**Hecho:**
- DNSMOS: leída la parte descriptiva del paper (arXiv:2110.01763) a partir de un resumen, para entender el funcionamiento general (estimación de MOS sin referencia limpia, submétricas SIG/BAK/OVRL).
- NISQA: mismo enfoque — resumen + lectura de la parte descriptiva del paper (arXiv:2104.09494), como comparativa con DNSMOS al ser de la misma familia (predictores neuronales de MOS no-intrusivos).

**Próximos pasos:**
- Profundizar en DNSMOS y NISQA antes de redactar Metodología (tratamiento en profundidad para DNSMOS y comparativo para NISQA). Con las 6 métricas ya repasadas al menos una vez, el siguiente paso natural es empezar a redactar la sección de Metodología.

---

## 2026-07-28 — Lectura del paper de DeepFilterNet

**Hecho:**
- Lectura del paper de DeepFilterNet (arquitectura DeepFilterNet2/3), apoyada con resúmenes para facilitar la digestión de las partes más densas.
- Sesión realizada desde otro dispositivo, usando Gemini como apoyo.

**Próximos pasos:**
- Continuar con el paper de HTDemucs (siguiente modelo en el orden de la Fase 2).

---

## 2026-07-29 — Lectura del paper de HTDemucs

**Hecho:**
- Lectura del paper de HTDemucs v4, apoyada con resúmenes para facilitar la digestión de las partes más densas.
- Sesión realizada desde otro dispositivo, usando Gemini como apoyo.

**Próximos pasos:**
- Con DeepFilterNet y HTDemucs ya revisados a nivel de paper, continuar con el resto de modelos (MP-SENet, AudioSR, VoiceFixer, ClearVoice/MossFormer2) y retomar el desarrollo de la app (Fase 4).

---

## 2026-07-31 — Primera versión funcional de la app (Gradio, categoría Denoising)

**Hecho:**
- Cambio de orden en el plan de trabajo: adelantar el desarrollo de la aplicación (Fase 4) antes de empezar a redactar la memoria, para tener ya una demo funcional y contenido real (comportamiento del pipeline, casos límite con audio real) de cara a la sección de diseño de la app y a los resultados.
- Diseño de la interfaz Gradio en dos niveles: Nivel 1 = categoría de degradación, Nivel 2 = modelo dentro de esa categoría (individual vs. combinado MossFormer2, cuando aplique).
- Gestión de memoria: un único modelo cargado en GPU a la vez, liberando el anterior antes de cargar el siguiente, generalizado en un `model_manager.py`.
- Primer wrapper de inferencia conectado: DeepFilterNet3 (denoising), con baseline clásico (spectral gating) y métricas DNSMOS sobre las tres versiones (original, IA, baseline).

**Problemas / bloqueos:**
1. Colisión de nombres de paquete: las carpetas `utils/` y `models/` colisionaban con paquetes genéricos ya presentes en el entorno de Colab. Solución: renombradas a `tfg_audio_utils/` y `tfg_models/`.
2. Orden de aplicación de los shims de torchaudio: el parche de compatibilidad se aplicaba dentro de la función de carga del modelo, pero un import de `df.enhance` en otra función se ejecutaba antes, rompiendo la importación. Solución: aplicar el shim a nivel de módulo.
3. Guardado incorrecto del audio procesado: se usaba el `guardar_audio` genérico (basado en `soundfile`) para la salida del modelo IA, cuando la notebook original usaba `save_audio` (de `df.enhance`). Causaba `LibsndfileError: Format not recognised`.

**Decisiones tomadas:**
- Al probar DeepFilterNet3 con el audio real del tutor (reverb fuerte, mic distante), el modelo generó picos de amplitud muy por encima de 1.0 (wraparound al convertir a int16). Solucionado normalizando el pico antes de guardar. DNSMOS puntuó la salida muy por encima del original y del baseline pese a la saturación perceptual evidente — segundo caso de discrepancia entre métrica no-intrusiva y percepción subjetiva, que refuerza el argumento central de la metodología dual del TFG.

**Próximos pasos:**
- Revisar si sigue habiendo un recorte de audio al inicio de la salida IA.
- Conectar el siguiente modelo (HTDemucs v4, separación de fuentes).

---

## 2026-08-02 — App funcional: depuración a fondo del pipeline de denoising

**Hecho:**
- Reorganización de notebooks: `01_DeepFilterNet_3.ipynb` vuelve a contener solo la validación individual del modelo, sin celdas de Gradio. Nueva notebook `app_gradio.ipynb` dedicada a instalar dependencias y lanzar la app.
- Fijada `gradio==6.20.0` en requirements y en la notebook de la app, tras detectar un `TypeError` reproducible en `gradio_client/utils.py` con las versiones 4.44.1 y 5.9.1.
- Resuelto el conflicto de numpy provocado por instalar Gradio (arrastraba pandas que exige numpy≥2, rompiendo la extensión nativa de deepfilternet): reinstalar `numpy==1.26.4` justo después de instalar Gradio, seguido de reinicio de runtime.
- Corregido el bug real de `LibsndfileError: Format not recognised`: DeepFilterNet3 devuelve el audio estéreo como `(canales, muestras)`, pero `soundfile.write()` espera `(muestras, canales)`. Fix en `tfg_models/denoising.py`: transponer (`.T`) solo cuando el array tiene 2 dimensiones.

**Estado final:**
- Categoría de Denoising (DeepFilterNet3) funcionando de extremo a extremo en la app.

---

## 2026-08-03 — App Gradio: MP-SENet (dereverberation) y VoiceFixer v2 (de-clipping) conectados

**Hecho:**
- Conectado MP-SENet a la app (paquete `MPSENet`, checkpoint `JacobLinCool/MP-SENet-VB`).
- Conectado VoiceFixer v2 a la app (API archivo-a-archivo, salida fija a 44100 Hz).
- Refactor de `app.py`: la selección de función de inferencia y baseline pasa de `if/elif` a dos diccionarios (`FUNCIONES_INFERENCIA`, `BASELINES_CLASICOS`).

**Problemas / bloqueos:**
- **Hallazgo importante (para la memoria):** con el audio real del tutor, la salida de MP-SENet suena a ruido blanco casi constante, con solo 2 fragmentos donde se entiende algo de voz — pero DNSMOS puntúa la IA por encima del baseline clásico. Causa raíz: el paquete pip `MPSENet` solo expone los checkpoints `g_best_dns`/`g_best_vb`, que son de **denoising** (DNS-Challenge/VoiceBank+DEMAND), no de dereverberation. La versión larga del repo original (yxlu-0102) sí cubre dereverb+BWE, pero no está empaquetada en el pip usado.
- Bug de configuración: `login(token=...)` de HF no persiste entre reinicios de kernel — hay que moverlo a la celda posterior al *último* reinicio, justo antes de lanzar la app.

**Decisiones tomadas:**
- No perseguir la instalación manual de la versión larga de MP-SENet por tiempo; se documenta como limitación y como el caso más severo de domain mismatch encontrado hasta ese momento.

**Próximos pasos:**
- Conectar HTDemucs v4 (separación de fuentes).

---

## 2026-08-04 — App Gradio: ClearVoice/MossFormer2 conectado. App completa (6 categorías)

**Hecho:**
- Conectado el último modelo pendiente, ClearVoice/MossFormer2 (combinado SE+SR), a las 3 categorías donde aplica (Denoising, Dereverberation, Super-resolución).
- Depuración larga de despliegue: los 504 Gateway Timeout repetidos en el túnel público de Gradio no eran por el modelo (que tarda ~26s con GPU), sino por (1) la primera descarga de checkpoints ocurriendo dentro de la ventana de la petición HTTP, y (2) servidores Gradio "zombis" acumulados por relanzar `launch()` sin `close()` de por medio.
- Solución aplicada: celda de precarga de MossFormer2 antes de lanzar Gradio, y protocolo de lanzamiento limpio (reiniciar sesión si hay dudas, lanzar `launch()` una sola vez, nunca relanzar sin `close()` antes).
- Probado `ngrok` como túnel alternativo: confirmó que el modelo respondía bien, pero el plan gratuito interceptaba las peticiones internas de Gradio con un interstitial de aviso, rompiendo la visualización del resultado. Se descarta por ahora; el túnel por defecto de Gradio (`share=True`) funciona siguiendo el protocolo de lanzamiento limpio.

**Problemas / bloqueos:**
- Hallazgo empírico: con un audio de prueba muy corto (~3s), el pipeline combinado SE+SR dio DNSMOS peor que el baseline clásico. MossFormer2 no trocea internamente (a diferencia de AudioSR), así que con clips muy cortos fuera del rango típico de entrenamiento puede comportarse de forma menos estable.

**Decisiones tomadas:**
- **App Gradio completada: las 6 categorías de degradación están conectadas.** Cierra la Fase 4 del proyecto.

**Próximos pasos:**
- Fase 5 — despliegue en Hugging Face Space (pendiente de decidir SDK Docker vs. Gradio simple).

---

## 2026-09-03 — Incidente de entorno en Colab (Python 3.12→3.13) y cambio de modelo de dereverberation

**Hecho:**
- Detectado que Google Colab actualizó, sin ningún cambio por parte del autor, la versión de Python por defecto de sus máquinas virtuales de 3.12 a 3.13, entre dos sesiones de trabajo.
- Esto rompió la instalación de varios paquetes clave del proyecto (`deepfilternet`, `audiosr`, `clearvoice`, y el nuevo SGMSE+ — ver más abajo), todos con una dependencia fijada a `numpy<2.0`, para la que ya no existen wheels precompiladas en Python 3.13, forzando una compilación desde código fuente inviable en tiempo razonable.
- Diagnosticado y corregido un bug de corte de audio en AudioSR: cada chunk se recomponía sin normalizar su longitud exacta de salida, lo que arrastraba una pérdida de cola de audio al final de los clips procesados. Fix: normalizar cada `chunk_out` a su `target_len` exacto (recorte o padding por repetición de borde) antes de concatenar.
- Diagnosticado un bug en el baseline clásico de BWE, que devolvía las mismas métricas que el audio original sin procesar (sospecha: `sr_origen == sr_destino` provocando un resample sin efecto sobre el audio de prueba usado).
- El túnel público de Gradio (`share=True`) volvió a dar problemas de estabilidad tras los cambios de entorno de esta sesión (desconexiones frecuentes durante las pruebas). Se retoma `ngrok` como túnel alternativo — esta vez sin el problema del interstitial detectado el 4 de agosto — y se adopta como solución definitiva para las sesiones de trabajo y prueba de la app.

**Problemas / bloqueos:**
- Todos resueltos durante la propia sesión.

**Decisiones tomadas:**
- Crear un entorno virtual aislado de Python 3.11 dentro de la propia máquina virtual de Colab (`/content/env311`), para el que sí existen wheels precompiladas de todas las dependencias del proyecto. De este modo, la aplicación y sus dependencias quedan desacopladas de la versión de Python que Colab decida ofrecer por defecto en cada sesión, evitando que un cambio de entorno ajeno al propio código vuelva a romper la instalación en el futuro.
- **Sustitución de MP-SENet por SGMSE+ como modelo activo de dereverberation en la aplicación.** Motivo: el paquete de PyPI de MP-SENet solo incluye checkpoints de denoising (limitación ya detectada el 3 de agosto), mientras que SGMSE+ sí está entrenado específicamente para dereverberation. MP-SENet se mantiene documentado en la memoria como caso de estudio de la propia limitación (divergencia entre métrica objetiva y calidad perceptual real).
- Se revierte parcialmente la decisión del 4 de agosto de descartar ngrok, tras confirmar que el problema del interstitial no vuelve a aparecer en esta configuración.

**Próximos pasos:**
- Confirmar el funcionamiento correcto de las 6 pipelines de la app con el nuevo entorno (venv311 + SGMSE+).
- Diseñar la metodología de testeo con audios sintéticos para reforzar el bloque empírico.

---

## 2026-09-04 — App verificada de punta a punta y diseño del barrido de audios sintéticos

**Hecho:**
- Confirmado el funcionamiento correcto de las 6 pipelines de la app sobre el nuevo entorno (venv311 + SGMSE+ + túnel ngrok).
- Diseñado el barrido de severidad por categoría para reforzar el bloque empírico con audios sintéticos, generados a partir de grabaciones limpias propias (narración en estudio de fragmentos de las primeras páginas de El Quijote): 6 niveles de SNR para denoising, 6 de T60 para dereverberation, 5 de frecuencia de corte para BWE, 6 de porcentaje de saturación para de-clipping, 6 de SIR para separación de fuentes.
- Diseñada una verificación de robustez adicional con un segundo fragmento del Quijote (2 niveles por categoría — uno moderado y el más severo), para comprobar que la forma de las curvas de severidad no depende del fragmento concreto.
- Diseñado el esquema del CSV de métricas a generar (una fila por combinación de categoría, fragmento, variante y nivel de severidad) y el diseño final del análisis a partir de él (curva de respuesta a la severidad + Δmétrica frente al audio sin procesar + marcador del valor bibliográfico publicado, combinados en una figura de dos paneles por categoría, más interpretación escrita conectando con el marco teórico).

**Problemas / bloqueos:**
- Ninguno.

**Decisiones tomadas:**
- No calcular baselines clásicos no-IA sobre el barrido de audios sintéticos: la comparativa IA vs. no-IA ya queda cubierta con los resultados obtenidos sobre el audio real del tutor (apartados 4.4.1-4.4.6). El barrido sintético compara únicamente el audio degradado sin procesar frente a la(s) variante(s) de IA de cada categoría, con el objetivo de habilitar métricas intrusivas (PESQ, STOI, SI-SDR, LSD) allí donde el audio real no dispone de referencia limpia.

**Próximos pasos:**
- Escribir el script de generación de los audios sintéticos degradados (`generar_audios_prueba.py`) y ponerlo a correr.

---

## 2026-09-05 — Generación de audios sintéticos y cómputo automático de métricas

**Hecho:**
- Escrito `generar_audios_prueba.py`: genera los audios degradados sintéticos en las 5 categorías, a partir de dos fragmentos limpios de referencia y un audio de interferencia (para separación de fuentes), aplicando el barrido de severidad diseñado el día anterior, y escribiendo un manifiesto CSV con los parámetros exactos de cada archivo generado.
- Puesto a punto un entorno de desarrollo en local (Windows/VS Code, entorno virtual de Python propio) para poder ejecutar este script fuera de Colab, ya que trabaja sobre archivos del propio ordenador.
- Corregido un cuello de botella de rendimiento crítico en la generación sintética de reverberación: la convolución de la señal con la respuesta al impulso se hacía con `numpy.convolve` (O(n·m), inviable para clips largos), sustituida por `scipy.signal.fftconvolve`.
- Generados los primeros 39 audios sintéticos degradados + manifiesto, sobre una carpeta propia sincronizada después con Drive.
- Escrito `calcular_metricas.py`: calcula automáticamente PESQ, STOI, SI-SDR, LSD y DNSMOS para todo el barrido, llamando directamente a las funciones de inferencia de la aplicación (`tfg_models`) desde Colab, sin pasar por la interfaz de Gradio ni copiar valores a mano.
- Detectados varios fallos de memoria GPU (CUDA out of memory) en el modelo de dereverberation durante el cómputo por lotes, al encadenar las distintas condiciones sin las pausas naturales del uso interactivo de la app. Añadida liberación explícita de memoria (`gc.collect()` + `torch.cuda.empty_cache()`) tras cada variante procesada, más un reintento automático único antes de dar una condición por fallida.
- Hecho `calcular_metricas.py` reanudable: si el CSV de resultados ya existe de una ejecución anterior, una nueva ejecución se salta automáticamente todo lo ya calculado y solo procesa lo que falte.

**Problemas / bloqueos:**
- Fallos de memoria GPU en dereverberation durante el cómputo por lotes (resueltos con la limpieza de memoria y el reintento automático, aunque no se pudieron descartar del todo hasta el día siguiente — ver 2026-09-06).

**Decisiones tomadas:**
- Ninguna adicional.

**Próximos pasos:**
- Revisar el CSV de métricas resultante en detalle antes de dar el barrido sintético por bueno.

---

## 2026-09-06 — Corrección del barrido sintético, cierre de la memoria y preparación del repositorio

**Hecho:**
- Revisado en profundidad el primer CSV de métricas generado. Detectados dos problemas: (1) la categoría de de-clipping producía audios prácticamente idénticos entre varios niveles de severidad consecutivos, y (2) faltaban por completo las filas del modelo de dereverberation para todas las condiciones.
- Diagnosticada la causa raíz común de ambos problemas: los fragmentos limpios de referencia (`audio1.wav`, `audio2.wav`) tenían en realidad una duración de casi 2 minutos, con solo unos 10 segundos de habla real al principio, en lugar de los 10 segundos previstos en el diseño del barrido. Esto diluía tanto el umbral de recorte de de-clipping (calculado por percentil de amplitud sobre toda la señal, incluidos los más de 100 segundos de silencio) como la potencia de referencia usada para dosificar ruido/interferencia en denoising y separación de fuentes, y multiplicaba por 12 el coste de memoria por condición en los modelos que no procesan por chunks, explicando también los fallos de memoria GPU del día anterior.
- Corregido `generar_audios_prueba.py` para recortar automáticamente los tres audios de entrada (los dos fragmentos limpios y el audio de interferencia) a sus primeros 10 segundos reales, sobrescribiéndolos en su propia carpeta, antes de generar ningún audio degradado. Regenerado el barrido completo de 39 audios sobre los fragmentos ya corregidos.
- Recalculado desde cero el CSV de métricas completo sobre los audios corregidos, esta vez sin fallos de memoria y sin el problema de valores idénticos en de-clipping.
- Revisada la memoria del TFG de principio a fin e incorporados los cambios pendientes: cláusula de audios sintéticos en Resumen (castellano/valenciano/inglés), Objetivos y Metodología; nuevo subapartado 4.3.1 con las incidencias técnicas del proyecto (cambio de versión de Python en Colab, sustitución de MP-SENet por SGMSE+, inestabilidad del túnel de Gradio y adopción de ngrok); corrección de una referencia cruzada rota y de una frase que quedaba cortada a mitad; conversión del epígrafe del modelo combinado a un encabezado numerado real; eliminación de la página de Resumen Ejecutivo con la tabla de competencias ABET, sin rellenar; nuevo apartado de resultados sobre el barrido de audios sintéticos degradados; y cierre de las Conclusiones incorporando las lecciones aprendidas del proyecto (adaptación del entorno de Colab, cambio de modelo de dereverberation, resolución del problema de Gradio/ngrok) junto con una síntesis final de los hallazgos más relevantes del trabajo.
- Preparado el repositorio de GitHub para su entrega: documentación de seguimiento (`objectives.md`, `TFG_plan.md`, `milestones.md`, `diary.md`) y `README.md` actualizados para reflejar el estado final del proyecto (SGMSE+ como modelo de dereverberation, sin despliegue como Hugging Face Space, con el bloque de audios sintéticos documentado), de modo que el tribunal pueda clonar el repositorio y reproducir la aplicación siguiendo las instrucciones del README.

**Problemas / bloqueos:**
- Todos los del día anterior, resueltos: la causa raíz (duración incorrecta de los audios de referencia) explica tanto el bug de de-clipping como los fallos de memoria GPU de dereverberation.

**Decisiones tomadas:**
- Descartar definitivamente el despliegue como Hugging Face Space por motivos de tiempo: la aplicación se ejecuta y se prueba en Google Colab (o en un entorno local equivalente), sin publicarse como Space. Se documenta en la memoria como una posible línea de trabajo futuro relacionada con la robustez del entorno de ejecución (ejecutar la app en un entorno local propio, no efímero), no como un objetivo incumplido.

**Próximos pasos:**
- Entrega de la memoria y del repositorio.
- Defensa del TFG ante el tribunal (fecha pendiente de confirmar).
