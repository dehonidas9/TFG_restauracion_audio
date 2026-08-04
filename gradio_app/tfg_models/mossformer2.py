"""
Wrapper de inferencia para la variante combinada multi-tarea: ClearVoice / MossFormer2.

A diferencia del resto de wrappers (que resuelven una única categoría), esta
variante puede seleccionarse desde el desplegable de Denoising, Dereverberation
o Super-resolución (BWE) indistintamente -- en los tres casos ejecuta el MISMO
pipeline combinado: primero MossFormer2_SE_48K (speech enhancement: denoising +
dereverberation) y después MossFormer2_SR_48K (super-resolución), tal y como se
validó en la notebook 06 (comparativa SE-alone vs. SE+SR). El baseline clásico
contra el que se compara sigue dependiendo de la categoría desde la que se
seleccione (denoising / dereverb / BWE), gestionado en app.py.

Paquete usado: `clearvoice` (PyPI, modelscope/ClearerVoice-Studio). Ambos
submodelos trabajan a 48kHz.

Nota importante (ya documentada como hallazgo empírico en notebook 06): en
audio real muy fuera de distribución, el pipeline completo SE+SR puede dar
peor DNSMOS que solo-SE, y los modelos de SR generativos pueden alucinar
artefactos de alta frecuencia que afectan más a la percepción subjetiva que
a las métricas no-intrusivas -- ver notas de metodología/resultados.

Gestión de memoria: al ser dos submodelos encadenados (no simultáneos), cada
uno pasa por `model_manager.obtener_modelo()` con su propia clave
("mossformer2_se" / "mossformer2_sr"), así que el manager libera SE antes de
cargar SR automáticamente, sin necesitar tener los dos en memoria a la vez.
"""

import numpy as np

from tfg_models import model_manager

SR_MODELO = 48000


def _cargar_mossformer2_se():
    from clearvoice import ClearVoice

    modelo = ClearVoice(task="speech_enhancement", model_names=["MossFormer2_SE_48K"])
    return {"modelo": modelo}


def _cargar_mossformer2_sr():
    from clearvoice import ClearVoice

    modelo = ClearVoice(task="speech_super_resolution", model_names=["MossFormer2_SR_48K"])
    return {"modelo": modelo}


def procesar(ruta_audio: str, ruta_salida: str):
    """
    Ejecuta el pipeline combinado MossFormer2 (SE -> SR) sobre un archivo de
    audio en disco y guarda el resultado final en ruta_salida.

    Returns:
        audio_mejorado_arr (np.ndarray): señal de salida a 48kHz, ya
            normalizada (pico 0.95).
        sr_modelo (int): 48000 Hz (ambos submodelos trabajan a 48kHz).
        pico_antes (float): pico máximo absoluto ANTES de normalizar, medido
            sobre la salida final (tras SR) -- mismo criterio que en el
            resto de wrappers.
    """
    from tfg_audio_utils.audio_utils_funcionescomunes import normalizar_pico, guardar_audio

    # --- Paso 1: Speech Enhancement (denoising + dereverberation) ---
    paquete_se = model_manager.obtener_modelo(
        "combinado", "mossformer2_se", _cargar_mossformer2_se
    )
    modelo_se = paquete_se["modelo"]
    audio_se = modelo_se(input_path=ruta_audio, online_write=False)

    ruta_intermedia = "/tmp/_mossformer2_se_intermedio.wav"
    modelo_se.write(audio_se, output_path=ruta_intermedia)

    # --- Paso 2: Super-resolución sobre la salida ya mejorada por SE ---
    paquete_sr = model_manager.obtener_modelo(
        "combinado", "mossformer2_sr", _cargar_mossformer2_sr
    )
    modelo_sr = paquete_sr["modelo"]
    audio_sr = modelo_sr(input_path=ruta_intermedia, online_write=False)

    audio_mejorado_arr = np.asarray(audio_sr).squeeze()

    pico_antes = np.max(np.abs(audio_mejorado_arr))
    audio_mejorado_arr = normalizar_pico(audio_mejorado_arr, pico_objetivo=0.95)

    guardar_audio(ruta_salida, audio_mejorado_arr, SR_MODELO)

    return audio_mejorado_arr, SR_MODELO, pico_antes
