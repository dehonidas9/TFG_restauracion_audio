# Milestones del TFG

Vista de alto nivel del proyecto: fases, entregables y estado. A diferencia de `diary.md`
(registro cronológico de lo que se hace día a día), este archivo se actualiza solo cuando
cambia el estado de un hito, no en cada sesión de trabajo.

Estados posibles: `PENDIENTE` · `EN CURSO` · `HECHO` · `BLOQUEADO` · `DESCARTADO`

---

## Fase 0 — Setup base del pipeline en Colab
**Estado:** HECHO

- [x] Notebook `00_Setup_Base.ipynb`: montaje de Drive, estructura de carpetas, `HF_HOME`.
- [x] Utilidades de audio, gestión de memoria GPU.
- [x] Métricas de referencia (PESQ, STOI, SI-SDR, LSD) y no intrusivas (DNSMOS).
- [x] Baselines clásicos no-IA por categoría.
- [x] Utils extraídos a `.py` en `utils/` (Drive: `Proyecto_Audio`).

**Fecha objetivo:** —
**Fecha real:** anterior a 2026-07-13

---

## Fase 2 — Notebooks de inferencia por modelo
**Estado:** HECHO (6 de 6 completados)

| # | Modelo | Categoría | Estado |
|---|--------|-----------|--------|
| 1 | DeepFilterNet2/3 | Denoising | HECHO |
| 2 | HTDemucs v4 | Separación de fuentes | HECHO |
| 3 | VoiceFixer v2 | De-clipping | HECHO |
| 4 | MP-SENet | Dereverberation (ver nota) | HECHO |
| 5 | AudioSR | Super-resolución (BWE) | HECHO |
| 6 | ClearVoice / MossFormer2 | Combinado (denoising+dereverb+SR) | HECHO |

> **Nota (2026-09-03):** el paquete de PyPI de MP-SENet solo distribuye checkpoints de
> denoising, sin soporte de dereverberation. En la aplicación final el modelo activo de
> dereverberation es **SGMSE+**; MP-SENet se mantiene documentado en la memoria como caso
> de estudio de esta misma limitación (divergencia entre métrica objetiva y calidad
> perceptual real).

**Fecha objetivo:** —
**Fecha real (1-5):** anterior a 2026-07-13 · **Fecha real (6):** 2026-08-04

---

## Fase 3 — Recopilación de métricas bibliográficas
**Estado:** HECHO

- [x] Recopiladas PESQ/STOI/SI-SDR/LSD publicados por cada modelo (bloque bibliográfico), a partir de papers/repos oficiales.
- [x] Verificadas cifras exactas contra papers/repos (no de memoria).
- [x] Distinguidas fuentes conceptuales vs. citables (IEEE) e incorporadas al marco teórico / estado del arte de la memoria.

**Fecha real:** 2026-07-16 a 2026-07-18 (estudio de métricas) e incorporación continuada durante la redacción.

---

## Fase 4 — Integración Gradio
**Estado:** HECHO

- [x] Interfaz de dos niveles: categoría → modelo/baseline.
- [x] Audio antes/después + espectrograma comparativo + métricas mostradas.
- [x] Las 6 categorías de degradación conectadas de extremo a extremo (denoising, dereverberation, BWE, de-clipping, separación de fuentes, combinado MossFormer2).
- [x] Verificado el funcionamiento completo de las 6 pipelines sobre el entorno definitivo (`env311` + SGMSE+ + túnel `ngrok`).

**Fecha real:** 2026-08-04 (app completa) · verificación final: 2026-09-04.

---

## Fase 5 — Despliegue en Hugging Face Space
**Estado:** DESCARTADO (por tiempo)

- [x] Decisión tomada el 2026-09-06: no se despliega la aplicación como Hugging Face Space por motivos de tiempo. La aplicación se ejecuta y se prueba en Google Colab (o en un entorno local equivalente).
- Se documenta en la memoria como posible línea de trabajo futuro relacionada con la robustez del entorno de ejecución (ejecutar la app en un entorno local propio, no efímero), no como un objetivo incumplido.

---

## Fase 6 — Barrido de audios sintéticos degradados
**Estado:** HECHO

- [x] Diseño del barrido de severidad por categoría (6 niveles SNR / T60 / % saturación / SIR, 5 niveles de frecuencia de corte para BWE) sobre dos fragmentos de referencia (narración propia de El Quijote), con verificación de robustez sobre un segundo fragmento.
- [x] `generar_audios_prueba.py`: generación de los audios degradados sintéticos + manifiesto CSV.
- [x] `calcular_metricas.py`: cómputo automático y reanudable de PESQ/STOI/SI-SDR/LSD/DNSMOS sobre todo el barrido, llamando directamente a `tfg_models`.
- [x] Corregido el bug de duración de los audios de referencia (recorte a los 10 s reales de habla) y regenerado el barrido completo (39 audios) y el CSV de métricas final sin fallos de memoria GPU ni valores duplicados en de-clipping.
- [x] Resultados incorporados a la memoria como nuevo apartado de resultados sobre el barrido sintético.

**Fecha real:** 2026-09-04 (diseño) a 2026-09-06 (corrección y cierre).

---

## Redacción de la memoria
**Estado:** HECHO (cierre final)

- [x] Resumen trilingüe (castellano/valenciano/inglés), incluyendo cláusula sobre el bloque de audios sintéticos.
- [x] Objetivos (8 objetivos numerados).
- [x] Introducción (1.1 Motivaciones, 1.2 Planteamiento, 1.3 Estructura).
- [x] Metodología, con cláusula sobre el bloque de audios sintéticos.
- [x] Desarrollo y resultados (marco teórico, estado del arte, diseño de la app, apartado 4.3.1 de incidencias técnicas, resultados sobre audio real y sobre el barrido sintético).
- [x] Conclusiones y trabajo futuro, incluyendo las lecciones aprendidas del proyecto (entorno de Colab, cambio de modelo de dereverberation, resolución de Gradio/ngrok) y una síntesis final de hallazgos.
- [x] Bibliografía (IEEE).
- [x] Anexos.

**Límite:** 75 páginas, Times New Roman 11pt.
**Fecha real de cierre:** 2026-09-06.

---

## Preparación del repositorio de GitHub
**Estado:** HECHO

- [x] Documentación de seguimiento (`objectives.md`, `TFG_plan.md`, `milestones.md`, `diary.md`) actualizada al estado final del proyecto.
- [x] `README.md` actualizado (modelo de dereverberation, sin despliegue como Space, bloque de audios sintéticos documentado).
- [x] Repositorio listo para que el tribunal pueda clonarlo y reproducir la aplicación siguiendo las instrucciones del README.

**Fecha real:** 2026-09-06.

---

### Próximo hito inmediato

Entrega de la memoria y del repositorio, y defensa del TFG ante el tribunal (fecha pendiente de confirmar).
