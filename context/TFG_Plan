# Plan de trabajo del TFG

**Proyecto:** Análisis y aplicación de modelos de aprendizaje profundo para la restauración interactiva de señales de audio degradadas.

Este documento recoge las fases del proyecto, su estado actual y los hitos (milestones) previstos. Se actualiza a medida que avanza el trabajo.

## Resumen del enfoque

- No se entrena ningún modelo: se usan modelos preentrenados de Hugging Face, con inferencia sobre Google Colab (GPU T4), en un entorno virtual aislado de Python 3.11 (`env311`) creado dentro de la propia máquina de Colab para blindar las dependencias frente a cambios de versión de Python ajenos al proyecto.
- Metodología mixta: bloque bibliográfico (métricas publicadas) + bloque empírico sobre audio real del tutor (métricas no intrusivas) + bloque de audios sintéticos degradados de forma controlada (habilita métricas intrusivas en ausencia de referencia limpia en el audio real).
- Interfaz final en Gradio, ejecutada y probada en Google Colab (o entorno local equivalente) con túnel `ngrok`. **No se despliega como Hugging Face Space** (descartado por tiempo; ver Fase 4 y Notas).

## Fases del proyecto

### Fase 0 — Configuración base ✅ COMPLETADA
- Notebook `00_Setup_Base.ipynb`: montaje de Drive, estructura de carpetas, `HF_HOME`, utilidades de audio, gestión de memoria GPU, métricas (PESQ/STOI/SI-SDR/LSD + DNSMOS), baselines clásicos por categoría.
- Utilidades extraídas a ficheros `.py` en `utils/` (renombrado después a `tfg_audio_utils/` y `tfg_models/` para evitar colisión con paquetes genéricos de Colab).

### Fase 1 — Modelos de inferencia individuales ✅ COMPLETADA
- [x] **DeepFilterNet2/3** (denoising) — `01_DeepFilterNet_3.ipynb` finalizado y depurado.
- [x] **HTDemucs v4** (separación de fuentes) — `02_HTDemucs.ipynb`, usando el paquete `demucs-infer` (no `demucs`, abandonado y limitado a torch <2.2).
- [x] **VoiceFixer v2** (de-clipping) — finalizado.
- [x] **MP-SENet** — finalizado; ver nota de sustitución más abajo.
- [x] **AudioSR** (super-resolución / BWE) — finalizado, con chunking de ~5 s y corrección posterior (2026-09-03) del recorte de cola en la recomposición de chunks.
- [x] **ClearVoice / MossFormer2** (modelo combinado multi-tarea) — finalizado el 2026-07-14; cierra la Fase 1.

Patrón por notebook: instalar dependencias específicas → importar utilidades compartidas → inferencia sobre audio de prueba → guardar en `outputs/` → calcular DNSMOS → ejecutar baseline clásico equivalente → liberar memoria GPU.

> **Cambio de modelo (2026-09-03):** el paquete de PyPI de MP-SENet solo distribuye
> checkpoints de denoising, sin soporte de dereverberation. Se sustituye por **SGMSE+**
> como modelo activo de dereverberation en la aplicación; MP-SENet se mantiene
> documentado en la memoria como caso de estudio de esta misma limitación.

### Fase 2 — Recopilación bibliográfica de métricas ✅ COMPLETADA
- Recopiladas las métricas publicadas (PESQ, STOI, SI-SDR, LSD) por cada modelo, a partir de sus papers/repos oficiales (estudio de métricas: 2026-07-16 a 2026-07-18).
- Distinguidas fuentes conceptuales de fuentes citables; cifras verificadas contra fuente original (no de memoria) e incorporadas al marco teórico / estado del arte de la memoria.

### Fase 3 — Interfaz Gradio ✅ COMPLETADA
- Diseño de interfaz de dos niveles: categoría de degradación → modelo/baseline.
- Visualización de audio antes/después, espectrograma comparativo y métricas.
- Las 6 categorías conectadas de extremo a extremo (cierre: 2026-08-04); verificación completa sobre el entorno definitivo (`env311` + SGMSE+ + `ngrok`) el 2026-09-04.

### Fase 4 — Despliegue como Hugging Face Space ❌ DESCARTADO (por tiempo)
- Decisión tomada el 2026-09-06: no se despliega la aplicación como Hugging Face Space por motivos de tiempo. La aplicación se ejecuta y se prueba en Google Colab (o en un entorno local equivalente), con túnel `ngrok` para las sesiones de prueba y demo.
- Se documenta en la memoria como posible línea de trabajo futuro relacionada con la robustez del entorno de ejecución (ejecutar la app en un entorno local propio y persistente, no efímero), no como un objetivo incumplido.

### Fase 5 — Barrido de audios sintéticos degradados ✅ COMPLETADA
- Diseñado y generado un barrido controlado de severidad por categoría (SNR, T60, frecuencia de corte, % de saturación, SIR) sobre dos fragmentos limpios de referencia (narración propia de El Quijote), con verificación de robustez sobre un segundo fragmento.
- `generar_audios_prueba.py` (generación + manifiesto) y `calcular_metricas.py` (cómputo automático y reanudable de métricas, llamando directamente a `tfg_models`).
- **Incidencia relevante (2026-09-06):** los fragmentos de referencia tenían casi 2 minutos de duración real (con solo ~10 s de habla), en lugar de los 10 s previstos, lo que diluía el umbral de de-clipping y la potencia de referencia de denoising/separación, y multiplicaba por 12 el coste de memoria GPU en los modelos sin chunking. Corregido recortando los tres audios de entrada a sus primeros 10 s reales antes de generar el barrido; regenerados los 39 audios y recalculado el CSV de métricas completo sin fallos ni duplicados.
- No se calculan baselines clásicos no-IA sobre este barrido: la comparativa IA vs. no-IA ya queda cubierta por los resultados sobre el audio real del tutor (apartados 4.4.1-4.4.6).

### Redacción de la memoria (en paralelo) ✅ CERRADA
- [x] Objetivos — redactado y finalizado.
- [x] Introducción (Motivaciones, Planteamiento, Estructura).
- [x] Metodología, incluyendo el bloque de audios sintéticos.
- [x] Desarrollo y resultados (marco teórico, estado del arte, diseño de la app, apartado 4.3.1 de incidencias técnicas, resultados sobre audio real y sobre el barrido sintético).
- [x] Conclusiones y trabajo futuro, con lecciones aprendidas y síntesis final.
- [x] Bibliografía (IEEE).
- [x] Anexos.

## Milestones

| Milestone | Descripción | Fecha objetivo | Estado |
|---|---|---|---|
| M0 | Setup base y utilidades completos | — | ✅ Completado |
| M1 | Notebook DeepFilterNet finalizado | — | ✅ Completado |
| M2 | Notebook HTDemucs finalizado | — | ✅ Completado |
| M3 | Resto de notebooks individuales finalizados (VoiceFixer, MP-SENet, AudioSR) | — | ✅ Completado |
| M4 | Notebook combinado ClearVoice/MossFormer2 finalizado | 2026-07-14 | ✅ Completado |
| M5 | Métricas bibliográficas recopiladas para los 6 modelos | 2026-07-18 | ✅ Completado |
| M6 | Interfaz Gradio funcional (6 categorías conectadas) | 2026-08-04 | ✅ Completado |
| M7 | Despliegue en Hugging Face Space | — | ❌ Descartado (por tiempo; ver Fase 4) |
| M7' | Barrido de audios sintéticos diseñado, generado y validado | 2026-09-06 | ✅ Completado |
| M8 | Borrador completo de la memoria | 2026-09-06 | ✅ Completado |
| M9 | Entrega y defensa del TFG | por definir con el tutor | ⏳ Pendiente |

## Notas y decisiones clave

- AudioSR ya no es estrictamente estado del arte (modelos como Vocos-BWE lo superan), pero se mantiene como referencia estándar; se reconoce como limitación en las conclusiones.
- Todos los modelos seleccionados se consideran académicamente defendibles; las limitaciones de vigencia se señalan en conclusiones, no descalifican la elección.
- MP-SENet (versión extendida) también soporta BWE y de-clipping, lo cual sirve para justificar su selección por versatilidad; en la aplicación final el modelo activo de dereverberation es SGMSE+ (ver Fase 1).
- Google Colab cambió de Python 3.12 a 3.13 sin aviso entre sesiones (2026-09-03), rompiendo la instalación de varias dependencias fijadas a `numpy<2.0`. Se mitigó creando un entorno virtual aislado de Python 3.11 (`env311`) dentro de la propia máquina de Colab, desacoplando el proyecto de la versión de Python que Colab ofrezca por defecto.
- El túnel público de Gradio (`share=True`) resultó inestable en varias sesiones; se adopta `ngrok` como túnel definitivo tras confirmar (2026-09-03) que el problema del interstitial detectado el 2026-08-04 no reaparece en la configuración actual.
- Despliegue como Hugging Face Space descartado definitivamente por tiempo (2026-09-06); ver Fase 4.
