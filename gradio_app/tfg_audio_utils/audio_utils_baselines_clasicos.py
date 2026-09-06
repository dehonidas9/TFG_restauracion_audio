
import numpy as np
import librosa
import scipy.signal as signal
from scipy.interpolate import CubicSpline


def baseline_denoising_spectral_gating(audio, sr, umbral_db=-20):
    """
    Baseline de denoising: 'spectral gating' clásico. Estima el ruido de fondo a partir de los
    frames más silenciosos y resta ese perfil espectral del resto de la señal.
    """
    stft = librosa.stft(audio)
    magnitud, fase = np.abs(stft), np.angle(stft)

    perfil_ruido = np.percentile(magnitud, 10, axis=1, keepdims=True)

    umbral = perfil_ruido * (10 ** (umbral_db / 20))
    magnitud_limpia = np.where(magnitud > umbral, magnitud - perfil_ruido, 0.0)
    magnitud_limpia = np.maximum(magnitud_limpia, 0.0)

    stft_limpio = magnitud_limpia * np.exp(1j * fase)
    return librosa.istft(stft_limpio, length=len(audio))


def baseline_dereverb_filtro_paso_alto(audio, sr, frecuencia_corte=100):
    """
    Baseline de dereverberation: filtro paso-alto simple.
    """
    sos = signal.butter(4, frecuencia_corte, btype='high', fs=sr, output='sos')
    return signal.sosfilt(sos, audio)


def baseline_bwe_interpolacion_spline(audio, sr_origen, sr_destino):
    """
    Baseline de super-resolución/BWE: upsampling clásico por interpolación spline cúbica.
    """
    return librosa.resample(audio, orig_sr=sr_origen, target_sr=sr_destino, res_type='fft')


def baseline_declipping_interpolacion_cubica(audio, umbral=0.99):
    """
    Baseline de de-clipping: interpolación cúbica sobre muestras saturadas.
    """
    audio_reparado = audio.copy()
    indices_clipeados = np.where(np.abs(audio) >= umbral)[0]

    if len(indices_clipeados) == 0:
        return audio_reparado

    indices_validos = np.where(np.abs(audio) < umbral)[0]
    if len(indices_validos) < 4:
        return audio_reparado

    spline = CubicSpline(indices_validos, audio[indices_validos])
    audio_reparado[indices_clipeados] = spline(indices_clipeados)
    return audio_reparado


def baseline_separacion_hpss(audio, sr):
    """
    Baseline de separación de fuentes: HPSS (Harmonic-Percussive Source Separation) clásico
    de librosa, vía median-filtering en el espectrograma (Fitzgerald, 2010).

    No es un separador de 4 stems como HTDemucs (voz/batería/bajo/resto): es un método clásico
    de procesado de señal que solo distingue entre componente armónico (más parecido a voz e
    instrumentos melódicos) y componente percusivo (más parecido a batería/ataques transitorios).
    Se usa aquí como referencia no-IA más cercana conceptualmente a la separación de fuentes,
    comparando su componente armónico contra el stem 'vocals' de HTDemucs.

    Returns:
        armonico (np.ndarray): componente armónico (proxy clásico de "voz/melodía").
        percusivo (np.ndarray): componente percusivo (proxy clásico de "batería/ritmo").
    """
    armonico, percusivo = librosa.effects.hpss(audio)
    return armonico, percusivo
