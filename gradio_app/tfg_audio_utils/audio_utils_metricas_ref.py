import numpy as np
import librosa
from pesq import pesq
from pystoi import stoi


def calcular_pesq(audio_referencia, audio_procesado, sr):
    """
    PESQ (Perceptual Evaluation of Speech Quality). Rango aproximado: -0.5 a 4.5 (más alto = mejor).
    Requiere sr de 8000 o 16000 Hz (la librería `pesq` reamplea internamente si hace falta, pero
    es recomendable pasar el audio ya a 16kHz para evitar errores).
    """
    modo = 'wb' if sr >= 16000 else 'nb'  # wideband si sr>=16kHz, narrowband si no
    return pesq(sr, audio_referencia, audio_procesado, modo)


def calcular_stoi(audio_referencia, audio_procesado, sr):
    """
    STOI (Short-Time Objective Intelligibility). Rango 0 a 1 (más alto = más inteligible).
    """
    return stoi(audio_referencia, audio_procesado, sr, extended=False)


def calcular_si_sdr(audio_referencia, audio_procesado):
    """
    SI-SDR (Scale-Invariant Signal-to-Distortion Ratio), en dB (más alto = mejor).
    Implementación manual (no requiere librería adicional).
    """
    referencia = audio_referencia - np.mean(audio_referencia)
    procesado = audio_procesado - np.mean(audio_procesado)

    # Proyección escalada de la señal procesada sobre la referencia
    alpha = np.dot(procesado, referencia) / (np.dot(referencia, referencia) + 1e-8)
    proyeccion = alpha * referencia
    ruido = procesado - proyeccion

    return 10 * np.log10((np.sum(proyeccion ** 2) + 1e-8) / (np.sum(ruido ** 2) + 1e-8))


def calcular_lsd(audio_referencia, audio_procesado, n_fft=1024):
    """
    LSD (Log-Spectral Distance), en dB (más bajo = mejor). Muy usada en papers de BWE/super-resolución
    para medir diferencias espectrales, especialmente en las frecuencias altas reconstruidas.
    """
    ref_stft = np.abs(librosa.stft(audio_referencia, n_fft=n_fft))
    proc_stft = np.abs(librosa.stft(audio_procesado, n_fft=n_fft))

    min_frames = min(ref_stft.shape[1], proc_stft.shape[1])
    ref_stft, proc_stft = ref_stft[:, :min_frames], proc_stft[:, :min_frames]

    log_ref = np.log10(np.maximum(ref_stft, 1e-8) ** 2)
    log_proc = np.log10(np.maximum(proc_stft, 1e-8) ** 2)

    distancia_por_frame = np.sqrt(np.mean((log_ref - log_proc) ** 2, axis=0))
    return np.mean(distancia_por_frame)
