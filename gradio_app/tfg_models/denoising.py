"""
Wrapper de inferencia para la categoría de denoising.

Variante activa: DeepFilterNet3 (individual). Reutiliza exactamente la lógica
de la notebook 01_DeepFilterNet_3.ipynb (shims de torchaudio, init_df(), enhance()).

Variante pendiente: MossFormer2 (combinado) — se añadirá cuando esté lista
la notebook 06.
"""

import sys
import types

import numpy as np

from tfg_models import model_manager


def _aplicar_shims_torchaudio():
    """
    Reconstruye `torchaudio.backend.common.AudioMetaData` y `torchaudio.info()`,
    eliminados en torchaudio >=2.9 (migración a torchcodec). Idéntico a lo usado
    en la notebook 01.
    """
    import torchaudio
    import soundfile as sf

    if not hasattr(torchaudio, "backend"):
        backend_module = types.ModuleType("torchaudio.backend")
        common_module = types.ModuleType("torchaudio.backend.common")

        class AudioMetaData:
            """Stub de compatibilidad: la clase real fue eliminada en
            torchaudio >=2.9. Solo se usa como type hint interno en
            deepfilternet, no afecta a la inferencia."""

            def __init__(
                self,
                sample_rate=None,
                num_frames=None,
                num_channels=None,
                bits_per_sample=None,
                encoding=None,
            ):
                self.sample_rate = sample_rate
                self.num_frames = num_frames
                self.num_channels = num_channels
                self.bits_per_sample = bits_per_sample
                self.encoding = encoding

        common_module.AudioMetaData = AudioMetaData
        backend_module.common = common_module
        sys.modules["torchaudio.backend"] = backend_module
        sys.modules["torchaudio.backend.common"] = common_module

    if not hasattr(torchaudio, "info"):
        def _info_shim(file, **kwargs):
            from torchaudio.backend.common import AudioMetaData

            datos = sf.info(str(file))
            return AudioMetaData(
                sample_rate=datos.samplerate,
                num_frames=datos.frames,
                num_channels=datos.channels,
                bits_per_sample=0,
                encoding=datos.subtype,
            )

        torchaudio.info = _info_shim


# Se aplica al importar este módulo (no dentro de una función), para garantizar
# que el parche esté puesto ANTES de cualquier import de df/torchaudio, sin
# depender de qué función se llame primero (init_df, enhance, load_audio...).
_aplicar_shims_torchaudio()


def _cargar_deepfilternet():
    from df.enhance import init_df

    modelo_dfn, df_state, _ = init_df()  # por defecto: DeepFilterNet3
    return {"modelo": modelo_dfn, "df_state": df_state}


def procesar(ruta_audio: str, ruta_salida: str):
    """
    Ejecuta inferencia de DeepFilterNet3 sobre un archivo de audio en disco
    y guarda el resultado en ruta_salida.

    Returns:
        audio_mejorado_arr (np.ndarray): señal de salida ya normalizada (pico 0.95),
            usada después para calcular DNSMOS. Forma (canales, muestras) o (muestras,).
        sr_modelo (int): sample rate del modelo (48000 Hz en DeepFilterNet3)
        pico_antes (float): pico máximo absoluto ANTES de normalizar, útil para
            detectar y registrar estos casos de desbordamiento en la memoria.
    """
    from df.enhance import enhance, load_audio

    from tfg_audio_utils.audio_utils_funcionescomunes import normalizar_pico, guardar_audio

    paquete = model_manager.obtener_modelo(
        "denoising", "deepfilternet", _cargar_deepfilternet
    )
    modelo_dfn, df_state = paquete["modelo"], paquete["df_state"]
    sr_modelo = df_state.sr()

    # DeepFilterNet tiene su propia función load_audio que ya resamplea al sr correcto
    audio_entrada, _ = load_audio(ruta_audio, sr=sr_modelo)
    audio_mejorado = enhance(modelo_dfn, df_state, audio_entrada)

    if hasattr(audio_mejorado, "detach"):
        audio_mejorado_arr = np.array(audio_mejorado.detach().cpu()).squeeze()
    else:
        audio_mejorado_arr = np.array(audio_mejorado).squeeze()

    pico_antes = np.max(np.abs(audio_mejorado_arr))
    audio_mejorado_arr = normalizar_pico(audio_mejorado_arr, pico_objetivo=0.95)

    # Si el audio es estéreo, DeepFilterNet lo devuelve como (canales, muestras),
    # pero soundfile espera (muestras, canales) para escribir multicanal.
    # Sin esto, sf.write no reconoce el formato ("Format not recognised").
    if audio_mejorado_arr.ndim == 2:
        audio_mejorado_arr_guardar = audio_mejorado_arr.T
    else:
        audio_mejorado_arr_guardar = audio_mejorado_arr


    guardar_audio(ruta_salida, audio_mejorado_arr_guardar, sr_modelo)

    return audio_mejorado_arr, sr_modelo, pico_antes