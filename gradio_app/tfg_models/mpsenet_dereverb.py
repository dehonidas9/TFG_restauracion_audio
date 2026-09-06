"""
Wrapper de inferencia para la categoría de dereverberation.

Variante activa: MP-SENet (individual), vía el paquete de inferencia `MPSENet`
(github.com/JacobLinCool/MPSENet), que empaqueta para inferencia los
checkpoints originales del autor de MP-SENet, publicados en Hugging Face.

Se usa el checkpoint "JacobLinCool/MP-SENet-VB" (entrenado sobre
VoiceBank+DEMAND), que es el mismo que reporta el PESQ de 3.50 citado en el
bloque bibliográfico de la memoria.

Nota sobre segmentación: por defecto, MPSENet trocea el audio de entrada en
segmentos de 2 segundos para evitar overflow de memoria (ver README del
paquete). Se deja el comportamiento por defecto por ser el más seguro en un
T4 de Colab; si en el futuro se detectan artefactos de discontinuidad entre
segmentos en el bloque empírico, se puede aumentar `segment_size` al llamar
al modelo.

Variante pendiente: MossFormer2 (combinado) — ya cubierta por la notebook 06,
pendiente de conectar en la app.
"""

import numpy as np
import torch

from tfg_models import model_manager

CHECKPOINT_MPSENET = "JacobLinCool/MP-SENet-VB"


def _cargar_mpsenet():
    from MPSENet import MPSENet

    device = "cuda" if torch.cuda.is_available() else "cpu"
    modelo = MPSENet.from_pretrained(CHECKPOINT_MPSENET).to(device)
    return {"modelo": modelo, "device": device}


def procesar(ruta_audio: str, ruta_salida: str):
    """
    Ejecuta inferencia de MP-SENet sobre un archivo de audio en disco y
    guarda el resultado en ruta_salida.

    Returns:
        audio_mejorado_arr (np.ndarray): señal de salida ya normalizada (pico 0.95).
        sr_modelo (int): sample rate propio del checkpoint MP-SENet.
        pico_antes (float): pico máximo absoluto ANTES de normalizar. Se
            mantiene el mismo criterio que en denoising.py por si aparece un
            caso de desbordamiento análogo al de DeepFilterNet3 (útil para
            el bloque empírico).
    """
    import librosa

    from tfg_audio_utils.audio_utils_funcionescomunes import normalizar_pico, guardar_audio

    paquete = model_manager.obtener_modelo(
        "dereverberation", "mpsenet", _cargar_mpsenet
    )
    modelo = paquete["modelo"]
    sr_modelo = modelo.sampling_rate

    # librosa.load resamplea directamente al sr propio del checkpoint
    audio_entrada, _ = librosa.load(ruta_audio, sr=sr_modelo)

    audio_mejorado, sr_modelo, _notacion = modelo(audio_entrada)

    if hasattr(audio_mejorado, "detach"):
        audio_mejorado_arr = np.array(audio_mejorado.detach().cpu()).squeeze()
    else:
        audio_mejorado_arr = np.asarray(audio_mejorado).squeeze()

    pico_antes = np.max(np.abs(audio_mejorado_arr))
    audio_mejorado_arr = normalizar_pico(audio_mejorado_arr, pico_objetivo=0.95)

    guardar_audio(ruta_salida, audio_mejorado_arr, sr_modelo)

    return audio_mejorado_arr, sr_modelo, pico_antes
