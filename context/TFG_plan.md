# Plan de trabajo del TFG

**Proyecto:** Análisis y aplicación de modelos de aprendizaje profundo para la restauración interactiva de señales de audio degradadas.

Este documento recoge las fases del proyecto, su estado actual y los hitos (milestones) previstos. Se actualiza a medida que avanza el trabajo.

## Resumen del enfoque

- No se entrena ningún modelo: se usan modelos preentrenados de Hugging Face, con inferencia sobre Google Colab (GPU T4).
- Metodología mixta: bloque bibliográfico (métricas publicadas) + bloque empírico (audio real del tutor, métricas no intrusivas).
- Interfaz final en Gradio, desplegada como Hugging Face Space.

## Fases del proyecto

### Fase 0 — Configuración base ✅ COMPLETADA
- Notebook `00_Setup_Base.ipynb`: montaje de Drive, estructura de carpetas, `HF_HOME`, utilidades de audio, gestión de memoria GPU, métricas (PESQ/STOI/SI-SDR/LSD + DNSMOS), baselines clásicos por categoría.
- Utilidades extraídas a ficheros `.py` en `utils/` dentro de `Proyecto_Audio`.

### Fase 1 — Modelos de inferencia individuales (en curso)
- [x] **DeepFilterNet2/3** (denoising) — `01_DeepFilterNet_3.ipynb` finalizado y depurado.
- [ ] **HTDemucs v4** (separación de fuentes) — `02_HTDemucs.ipynb` (siguiente notebook). Nota: usar el paquete `demucs-infer`, no `demucs` (abandonado, requiere torch <2.2).
- [ ] **VoiceFixer v2** (de-clipping)
- [ ] **MP-SENet** (dereverberation)
- [ ] **AudioSR** (super-resolución / BWE)
- [ ] **ClearVoice / MossFormer2** (modelo combinado multi-tarea)

Patrón por notebook: instalar dependencias específicas → importar utilidades compartidas → inferencia sobre audio de prueba → guardar en `outputs/` → calcular DNSMOS → ejecutar baseline clásico equivalente → liberar memoria GPU.

### Fase 2 — Recopilación bibliográfica de métricas
- Recopilación en paralelo de las métricas publicadas (PESQ, STOI, SI-SDR, LSD) por cada modelo, a partir de sus papers/repos oficiales.
- Distinguir fuentes conceptuales de fuentes citables; verificar cifras exactas (no confiar en memoria).

### Fase 3 — Interfaz Gradio
- Diseño de interfaz de dos niveles: categoría de degradación → modelo/baseline.
- Visualización de audio antes/después, espectrograma comparativo y métricas.

### Fase 4 — Despliegue como Hugging Face Space
- Exportar como `app.py` + `requirements.txt`.
- Trasladar del entorno Colab al Space: mantener el toolchain de Rust en el Dockerfile y los shims de `torchaudio` en `app.py`; el pin de numpy y el reinicio de kernel son artefactos exclusivos de Colab y no deben trasladarse.
- Ver notas detalladas en `NOTAS_DESPLIEGUE_FASE5.md`.

### Redacción de la memoria (en paralelo)
- [x] Objetivos — redactado y finalizado.
- [ ] Introducción (Motivaciones, Planteamiento, Estructura)
- [ ] Metodología
- [ ] Desarrollo y resultados (marco teórico, estado del arte, diseño de la app, resultados)
- [ ] Conclusiones y trabajo futuro
- [ ] Bibliografía (IEEE)
- [ ] Anexos

## Milestones

| Milestone | Descripción | Fecha objetivo | Estado |
|---|---|---|---|
| M0 | Setup base y utilidades completos | — | ✅ Completado |
| M1 | Notebook DeepFilterNet finalizado | — | ✅ Completado |
| M2 | Notebook HTDemucs finalizado | *(por definir)* | ⏳ Pendiente |
| M3 | Resto de notebooks individuales finalizados (VoiceFixer, MP-SENet, AudioSR) | *(por definir)* | ⏳ Pendiente |
| M4 | Notebook combinado ClearVoice/MossFormer2 finalizado | *(por definir)* | ⏳ Pendiente |
| M5 | Métricas bibliográficas recopiladas para los 6 modelos | *(por definir)* | ⏳ Pendiente |
| M6 | Interfaz Gradio funcional | *(por definir)* | ⏳ Pendiente |
| M7 | Despliegue en Hugging Face Space | *(por definir)* | ⏳ Pendiente |
| M8 | Borrador completo de la memoria | *(por definir)* | ⏳ Pendiente |
| M9 | Entrega y defensa del TFG | *(por definir)* | ⏳ Pendiente |

> Rellenar las fechas objetivo con el tutor para llevar el seguimiento acordado.

## Notas y decisiones clave

- AudioSR ya no es estrictamente estado del arte (modelos como Vocos-BWE lo superan), pero se mantiene como referencia estándar; se reconocerá como limitación en las conclusiones.
- Todos los modelos seleccionados se consideran académicamente defendibles; las limitaciones de vigencia se señalan en conclusiones, no descalifican la elección.
- MP-SENet (versión extendida) también soporta BWE y de-clipping, lo cual sirve para justificar su selección por versatilidad.
