"""
App de restauración interactiva de audio degradado — TFG ETSIT/UPV.

Diseño de interfaz en dos niveles:
  Nivel 1 — Categoría de degradación (denoising, dereverb, BWE, de-clipping, separación)
  Nivel 2 — Modelo dentro de esa categoría (individual vs. combinado MossFormer2,
            cuando aplique)

Gestión de memoria: un único modelo cargado a la vez (ver models/model_manager.py),
siguiendo el mismo patrón que las notebooks de Colab (liberar_memoria_gpu entre
modelos), priorizando evitar problemas de VRAM sobre la velocidad.

Nota de diseño (añadida al conectar MP-SENet): la selección de función de
inferencia y de baseline clásico ahora se hace por diccionario
(FUNCIONES_INFERENCIA / BASELINES_CLASICOS) en vez de if/elif hardcodeado,
para que conectar AudioSR, VoiceFixer y HTDemucs más adelante sea solo
añadir una entrada, sin tocar la lógica de procesar_audio().
"""

import os
import sys

import gradio as gr
import pandas as pd

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from tfg_audio_utils.audio_utils_funcionescomunes import cargar_audio, guardar_audio
from tfg_audio_utils.audio_utils_metricas_no_intrusivas import calcular_dnsmos
from tfg_audio_utils.audio_utils_baselines_clasicos import (
    baseline_denoising_spectral_gating,
    baseline_dereverb_filtro_paso_alto,
    baseline_declipping_interpolacion_cubica,
)
from tfg_models import denoising, dereverb, declipping

# ---------------------------------------------------------------------------
# Categorías y variantes (Nivel 1 / Nivel 2)
# ---------------------------------------------------------------------------
CATEGORIAS = {
    "Denoising": {
        "activa": True,
        "variantes": {
            "DeepFilterNet3 (individual)": "deepfilternet",
            "MossFormer2 (combinado) — próximamente": None,
        },
    },
    "Dereverberation": {
        "activa": True,
        "variantes": {
            "MP-SENet (individual)": "mpsenet",
            "MossFormer2 (combinado) — próximamente": None,
        },
    },
    "Super-resolución (BWE)": {
        "activa": False,
        "variantes": {
            "AudioSR (individual) — próximamente": None,
            "MossFormer2 (combinado) — próximamente": None,
        },
    },
    "De-clipping": {
        "activa": True,
        "variantes": {"VoiceFixer v2 (individual)": "voicefixer"},
    },
    "Separación de fuentes": {
        "activa": False,
        "variantes": {"HTDemucs v4 — próximamente": None},
    },
}

# clave_variante -> función procesar(ruta_audio, ruta_salida) -> (audio, sr, pico_antes)
FUNCIONES_INFERENCIA = {
    "deepfilternet": denoising.procesar,
    "mpsenet": dereverb.procesar,
    "voicefixer": declipping.procesar,
}

# categoria -> función baseline(audio_original, sr_original) -> audio_baseline
BASELINES_CLASICOS = {
    "Denoising": baseline_denoising_spectral_gating,
    "Dereverberation": baseline_dereverb_filtro_paso_alto,
    # baseline_declipping_interpolacion_cubica no recibe sr (solo audio y un
    # umbral con default 0.99), así que se envuelve para respetar la firma
    # común baseline(audio_original, sr_original) que usa procesar_audio().
    "De-clipping": lambda audio, sr: baseline_declipping_interpolacion_cubica(audio),
}


def actualizar_variantes(categoria: str):
    opciones = list(CATEGORIAS[categoria]["variantes"].keys())
    return gr.Dropdown(choices=opciones, value=opciones[0])


def procesar_audio(ruta_audio, categoria, variante_label):
    if ruta_audio is None:
        raise gr.Error("Sube un archivo de audio antes de procesar.")

    clave_variante = CATEGORIAS[categoria]["variantes"][variante_label]
    if not CATEGORIAS[categoria]["activa"] or clave_variante is None:
        raise gr.Error(
            f"'{variante_label}' en la categoría '{categoria}' todavía no está "
            "implementado en esta app."
        )

    funcion_inferencia = FUNCIONES_INFERENCIA.get(clave_variante)
    if funcion_inferencia is None:
        raise gr.Error(f"Variante '{clave_variante}' sin wrapper de inferencia todavía.")

    funcion_baseline = BASELINES_CLASICOS.get(categoria)
    if funcion_baseline is None:
        raise gr.Error(
            f"Categoría '{categoria}' todavía no tiene baseline clásico conectado."
        )

    # --- Cargar audio original ---
    audio_original, sr_original = cargar_audio(
        ruta_audio, sr_objetivo=None, forzar_mono=True
    )

    # --- Inferencia IA ---
    ruta_salida_ia = "/tmp/salida_ia.wav"
    audio_mejorado, sr_modelo, pico_antes = funcion_inferencia(ruta_audio, ruta_salida_ia)

    aviso_saturacion = ""
    if pico_antes > 1.0:
        aviso_saturacion = (
            f"\n\n⚠️ Aviso: el modelo generó un pico de {pico_antes:.2f} "
            "(por encima de 1.0) en este audio; se ha normalizado antes de "
            "guardar para evitar distorsión por desbordamiento. Esto suele "
            "indicar que el audio de entrada está muy alejado del dominio "
            "de entrenamiento del modelo (útil para el bloque empírico)."
        )

    # --- Baseline clásico (siempre se calcula, para la comparativa IA vs. no-IA) ---
    audio_baseline = funcion_baseline(audio_original, sr_original)

    # --- Guardar baseline para reproducir/descargar en la interfaz ---
    ruta_salida_baseline = "/tmp/salida_baseline.wav"
    guardar_audio(ruta_salida_baseline, audio_baseline, sr_original)

    # --- Métricas DNSMOS (no intrusivas, sin necesidad de referencia limpia) ---
    dnsmos_original = calcular_dnsmos(audio_original, sr_original)
    dnsmos_ia = calcular_dnsmos(audio_mejorado, sr_modelo)
    dnsmos_baseline = calcular_dnsmos(audio_baseline, sr_original)

    tabla = pd.DataFrame(
        [
            {
                "Versión": "Original (degradado)",
                "OVRL": dnsmos_original.get("ovrl_mos"),
                "SIG": dnsmos_original.get("sig_mos"),
                "BAK": dnsmos_original.get("bak_mos"),
            },
            {
                "Versión": f"{variante_label} (IA)",
                "OVRL": dnsmos_ia.get("ovrl_mos"),
                "SIG": dnsmos_ia.get("sig_mos"),
                "BAK": dnsmos_ia.get("bak_mos"),
            },
            {
                "Versión": "Baseline (no-IA)",
                "OVRL": dnsmos_baseline.get("ovrl_mos"),
                "SIG": dnsmos_baseline.get("sig_mos"),
                "BAK": dnsmos_baseline.get("bak_mos"),
            },
        ]
    )

    return ruta_salida_ia, ruta_salida_baseline, tabla, aviso_saturacion


# ---------------------------------------------------------------------------
# Interfaz Gradio
# ---------------------------------------------------------------------------
with gr.Blocks(title="Restauración interactiva de audio — TFG ETSIT/UPV") as demo:
    gr.Markdown(
        "# Restauración interactiva de señales de audio degradadas\n"
        "TFG — ETSIT/UPV · Análisis y aplicación de modelos de aprendizaje profundo"
    )

    with gr.Row():
        categoria = gr.Dropdown(
            label="1. Categoría de degradación",
            choices=list(CATEGORIAS.keys()),
            value="Denoising",
        )
        variante = gr.Dropdown(
            label="2. Modelo",
            choices=list(CATEGORIAS["Denoising"]["variantes"].keys()),
            value=list(CATEGORIAS["Denoising"]["variantes"].keys())[0],
        )

    categoria.change(actualizar_variantes, inputs=categoria, outputs=variante)

    audio_entrada = gr.Audio(label="Audio de entrada", type="filepath")
    boton = gr.Button("Procesar", variant="primary")

    with gr.Row():
        salida_ia = gr.Audio(label="Resultado — Modelo IA")
        salida_baseline = gr.Audio(label="Resultado — Baseline clásico (no-IA)")

    tabla_metricas = gr.Dataframe(
        headers=["Versión", "OVRL", "SIG", "BAK"],
        datatype=["str", "number", "number", "number"],
        label="DNSMOS (métrica no intrusiva, sin referencia limpia)",
    )
    aviso = gr.Markdown()

    boton.click(
        procesar_audio,
        inputs=[audio_entrada, categoria, variante],
        outputs=[salida_ia, salida_baseline, tabla_metricas, aviso],
    )

if __name__ == "__main__":
    demo.queue()
    demo.launch(server_name="0.0.0.0", server_port=7860, show_api=False)