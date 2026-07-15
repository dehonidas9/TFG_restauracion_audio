# Milestones del TFG

Vista de alto nivel del proyecto: fases, entregables y estado. A diferencia de `diary.md`
(registro cronológico de lo que se hace día a día), este archivo se actualiza solo cuando
cambia el estado de un hito, no en cada sesión de trabajo.

Estados posibles: `PENDIENTE` · `EN CURSO` · `HECHO` · `BLOQUEADO`

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
**Estado:** EN CURSO (5 de 6 completados)

| # | Modelo | Categoría | Estado |
|---|--------|-----------|--------|
| 1 | DeepFilterNet2/3 | Denoising | HECHO |
| 2 | HTDemucs v4 | Separación de fuentes | HECHO |
| 3 | VoiceFixer v2 | De-clipping | HECHO |
| 4 | MP-SENet | Dereverberation | HECHO |
| 5 | AudioSR | Super-resolución (BWE) | HECHO |
| 6 | ClearVoice / MossFormer2 | Combinado (denoising+dereverb+SR) | **PENDIENTE — siguiente** |

**Fecha objetivo:** —
**Fecha real (1-5):** anterior a 2026-07-13

---

## Fase 3 — Recopilación de métricas bibliográficas
**Estado:** PENDIENTE

- [ ] Recopilar PESQ/STOI/SI-SDR/LSD publicados por cada modelo (bloque bibliográfico).
- [ ] Verificar cifras exactas contra papers/repos (no de memoria).
- [ ] Distinguir fuentes conceptuales vs. citables (IEEE).

**Fecha objetivo:** por definir con el tutor

---

## Fase 4 — Integración Gradio
**Estado:** PENDIENTE

- [ ] Interfaz de dos niveles: categoría → modelo/baseline.
- [ ] Audio antes/después + espectrograma comparativo + métricas mostradas.

**Fecha objetivo:** por definir con el tutor

---

## Fase 5 — Despliegue en Hugging Face Space
**Estado:** PENDIENTE

- [ ] Exportar como `app.py` + `requirements.txt`.
- [ ] Trasladar al Dockerfile/`app.py` lo necesario (toolchain Rust, shims de torchaudio), sin arrastrar artefactos exclusivos de Colab (pin de numpy, reinicio de kernel).

**Fecha objetivo:** por definir con el tutor

---

## Redacción de la memoria
**Estado:** EN CURSO

- [x] Objetivos (8 objetivos numerados) — finalizado.
- [ ] Resumen trilingüe (castellano/valenciano/inglés).
- [ ] Introducción (1.1 Motivaciones, 1.2 Planteamiento, 1.3 Estructura).
- [ ] Metodología.
- [ ] Desarrollo y resultados (marco teórico, estado del arte, diseño de la app, resultados).
- [ ] Conclusiones y trabajo futuro.
- [ ] Bibliografía (IEEE).
- [ ] Anexos.

**Límite:** 75 páginas, Times New Roman 11pt.
**Fecha objetivo de entrega:** por definir con el tutor


Próximo hito inmediato
Fase 3 — Recopilación de métricas bibliográficas (PESQ, STOI, SI-SDR, LSD publicados por cada uno de los 6 modelos), y arranque de la redacción de Metodología / Desarrollo y resultados incorporando el cambio de patrón del notebook 6 y el hallazgo del DNSMOS.
