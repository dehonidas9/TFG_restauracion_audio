"""
Wrapper de inferencia para la categoría de separación de fuentes.

Variante activa: HTDemucs v4, vía el paquete `demucs-infer` (fork de solo
inferencia de Demucs, compatible con PyTorch 2.x). Se usa este fork en vez
del paquete `demucs` original porque este último fija `torchaudio<2.2`, lo
que entraría en conflicto con la versión moderna de torchaudio que ya
necesitan DeepFilterNet3 y el resto de modelos de la app (el mismo motivo
por el que hubo que aplicar shims de compatibilidad en denoising.py).

Diferencia de diseño importante respecto al resto de wrappers: HTDemucs no
devuelve una única señal "restaurada", sino 4 stems (drums, bass, other,
vocals). Para mantener la misma interfaz "salida IA vs. baseline" que usa
el resto de la app, se usa el stem 'vocals' como salida IA -- comparándolo
contra el componente armónico de HPSS (ver baseline_separacion_hpss en
audio_utils_baselines_clasicos.py, que ya documenta esta comparación como
la más cercana conceptualmente entre un método clásico y un separador de
4 stems).

Requisito importante: HTDemucs espera entrada ESTÉREO (2 canales). Si el
audio de entrada es mono, se duplica al canal restante antes de separar.

Variante pendiente: ninguna (Separación de fuentes solo tiene HTDemucs en
el TFG).
"""

import numpy as np
import torch
import torchaudio

from tfg_models import model_manager

NOMBRE_MODELO = "htdemucs"


def _cargar_htdemucs():
    from demucs_infer.pretrained import get_model

    device = "cuda" if torch.cuda.is_available() else "cpu"
    modelo = get_model(NOMBRE_MODELO)
    modelo.to(device)
    modelo.eval()
    return {"modelo": modelo, "device": device}


def procesar(ruta_audio: str, ruta_salida: str):
    """
    Ejecuta separación de fuentes con HTDemucs sobre un archivo de audio en
    disco y guarda el stem 'vocals' en ruta_salida.

    Returns:
        audio_mejorado_arr (np.ndarray): stem 'vocals' en mono, ya
            normalizado (pico 0.95).
        sr_modelo (int): sample rate propio de HTDemucs (model.samplerate).
        pico_antes (float): pico máximo absoluto ANTES de normalizar.
    """
    from demucs_infer.apply import apply_model

    from tfg_audio_utils.audio_utils_funcionescomunes import normalizar_pico, guardar_audio

    paquete = model_manager.obtener_modelo(
        "separacion_fuentes", "htdemucs", _cargar_htdemucs
    )
    modelo, device = paquete["modelo"], paquete["device"]
    sr_modelo = modelo.samplerate

    wav, sr_entrada = torchaudio.load(ruta_audio)
    if sr_entrada != sr_modelo:
        wav = torchaudio.functional.resample(wav, sr_entrada, sr_modelo)

    # HTDemucs espera estéreo; si el audio es mono, se duplica el canal.
    if wav.shape[0] == 1:
        wav = wav.repeat(2, 1)

    wav = wav.unsqueeze(0)  # añade dimensión de batch

    with torch.no_grad():
        fuentes = apply_model(modelo, wav, device=device)

    indice_vocals = modelo.sources.index("vocals")
    vocals = fuentes[0, indice_vocals].cpu()

    audio_mejorado_arr = vocals.numpy().squeeze()
    # Se pasa a mono para mantener coherencia con el resto de wrappers
    # (que devuelven señales mono) y con el cálculo de DNSMOS.
    if audio_mejorado_arr.ndim == 2:
        audio_mejorado_arr = audio_mejorado_arr.mean(axis=0)

    pico_antes = np.max(np.abs(audio_mejorado_arr))
    audio_mejorado_arr = normalizar_pico(audio_mejorado_arr, pico_objetivo=0.95)

    guardar_audio(ruta_salida, audio_mejorado_arr, sr_modelo)

    return audio_mejorado_arr, sr_modelo, pico_antes
