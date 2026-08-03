"""
Wrapper de inferencia para la categoría de de-clipping.

Variante activa: VoiceFixer v2 (paquete `voicefixer`, haoheliu/voicefixer).

Diferencia clave respecto a denoising.py y dereverb.py: la API de VoiceFixer
es archivo-a-archivo (`restore(input=..., output=...)`) y no expone el audio
como array en memoria. Por eso aquí se escribe primero a un archivo temporal,
se relee con cargar_audio() para poder normalizar y calcular DNSMOS igual que
con el resto de wrappers, y solo entonces se guarda en la ruta final.

VoiceFixer produce SIEMPRE la salida a 44100 Hz (vocoder universal 44.1kHz),
independientemente del sample rate de entrada.

Modos de inferencia del propio paquete:
  0 -> modelo original (recomendado por defecto, el que se usa aquí)
  1 -> añade un preprocesado que recorta altas frecuencias
  2 -> "modo entrenamiento", que el propio autor describe como potencialmente
       útil en voz real muy degradada pero menos estable
Si mode=0 no da buen resultado con el audio real del tutor, probar mode=1
antes de descartar el modelo para el bloque empírico.

Variante pendiente: ninguna (De-clipping solo tiene esta variante en el TFG).
"""

import numpy as np
import torch

from tfg_models import model_manager


def _cargar_voicefixer():
    from voicefixer import VoiceFixer

    modelo = VoiceFixer()
    return {"modelo": modelo, "usa_gpu": torch.cuda.is_available()}


def procesar(ruta_audio: str, ruta_salida: str, mode: int = 0):
    """
    Ejecuta inferencia de VoiceFixer sobre un archivo de audio en disco y
    guarda el resultado en ruta_salida.

    Returns:
        audio_mejorado_arr (np.ndarray): señal de salida ya normalizada (pico 0.95).
        sr_modelo (int): 44100 Hz, fijo por el vocoder de VoiceFixer.
        pico_antes (float): pico máximo absoluto ANTES de normalizar (mismo
            criterio que en denoising.py/dereverb.py).
    """
    from tfg_audio_utils.audio_utils_funcionescomunes import (
        cargar_audio,
        normalizar_pico,
        guardar_audio,
    )

    paquete = model_manager.obtener_modelo(
        "declipping", "voicefixer", _cargar_voicefixer
    )
    modelo = paquete["modelo"]

    # restore() escribe directamente a disco; se usa un archivo temporal
    # intermedio para poder normalizar antes de guardar en ruta_salida.
    ruta_temporal = ruta_salida + ".voicefixer_tmp.wav"
    modelo.restore(
        input=ruta_audio,
        output=ruta_temporal,
        cuda=paquete["usa_gpu"],
        mode=mode,
    )

    sr_modelo = 44100
    audio_mejorado_arr, _ = cargar_audio(
        ruta_temporal, sr_objetivo=sr_modelo, forzar_mono=True
    )

    pico_antes = np.max(np.abs(audio_mejorado_arr))
    audio_mejorado_arr = normalizar_pico(audio_mejorado_arr, pico_objetivo=0.95)

    guardar_audio(ruta_salida, audio_mejorado_arr, sr_modelo)

    return audio_mejorado_arr, sr_modelo, pico_antes
