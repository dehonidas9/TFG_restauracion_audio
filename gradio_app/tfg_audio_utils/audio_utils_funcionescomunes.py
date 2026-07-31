import librosa
import soundfile as sf
import numpy as np


def cargar_audio(ruta, sr_objetivo=None, forzar_mono=True):
    """
    Carga un archivo de audio desde disco.

    Args:
        ruta (str): ruta al archivo de audio (.wav, .mp3, etc.)
        sr_objetivo (int, opcional): si se especifica, resamplea al cargar.
                                     Si es None, se mantiene el sample rate original.
        forzar_mono (bool): si True, hace downmix a mono (promedio de canales).

    Returns:
        audio (np.ndarray): señal de audio como array 1D (mono) o 2D (canales, muestras).
        sr (int): sample rate real del audio devuelto.
    """
    audio, sr = librosa.load(ruta, sr=sr_objetivo, mono=forzar_mono)
    return audio, sr


def guardar_audio(ruta_salida, audio, sr):
    """
    Guarda una señal de audio en disco como .wav.

    Args:
        ruta_salida (str): ruta de destino, debe terminar en .wav
        audio (np.ndarray): señal de audio (mono o estéreo)
        sr (int): sample rate de la señal
    """
    sf.write(ruta_salida, audio, sr)
    print(f'Audio guardado en: {ruta_salida}')


def resamplear(audio, sr_origen, sr_destino):
    """
    Resamplea una señal de audio de sr_origen a sr_destino.
    Necesario porque cada modelo trabaja a una frecuencia distinta
    (ej. DeepFilterNet a 48kHz, MP-SENet a 16kHz, AudioSR variable).
    """
    if sr_origen == sr_destino:
        return audio
    return librosa.resample(audio, orig_sr=sr_origen, target_sr=sr_destino)


def normalizar_pico(audio, pico_objetivo=0.95):
    """
    Normaliza el audio para que su valor de pico absoluto sea pico_objetivo.
    Útil antes de aplicar clipping sintético o antes de pasar por modelos
    sensibles al nivel de entrada.
    """
    pico_actual = np.max(np.abs(audio))
    if pico_actual == 0:
        return audio
    return audio * (pico_objetivo / pico_actual)
