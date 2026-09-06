import numpy as np
import librosa
from speechmos import dnsmos

SR_DNSMOS = 16000  # DNSMOS solo acepta audio a 16 kHz


def calcular_dnsmos(audio, sr):
    """
    DNSMOS (Deep Noise Suppression MOS). Devuelve una estimación de MOS (Mean Opinion Score)
    sin necesidad de referencia limpia. Ideal para el audio real del tutor.

    Devuelve un diccionario con submétricas: SIG (calidad de la señal de voz),
    BAK (calidad del ruido de fondo), OVRL (calidad general).

    Nota: DNSMOS exige que el audio esté a 16000 Hz y sea un array 1D (mono).
    Se resamplea automáticamente si hace falta, y si llega con más de un canal
    se promedia a mono antes de calcular la métrica.
    """
    audio = np.asarray(audio, dtype=np.float32)
    audio = np.squeeze(audio)

    if audio.ndim > 1:
        audio = audio.mean(axis=0)

    audio = np.ascontiguousarray(audio.reshape(-1))

    if sr != SR_DNSMOS:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=SR_DNSMOS)

    resultado = dnsmos.run(audio, sr=SR_DNSMOS)
    return resultado


# NISQA requiere clonar su repo oficial (no está empaquetado en pip de forma estable):
# !git clone https://github.com/gabrielmittag/NISQA.git
# Se deja como paso pendiente para el notebook de métricas empíricas, ya que necesita
# descargar sus propios pesos preentrenados (no vía HF_HOME).
