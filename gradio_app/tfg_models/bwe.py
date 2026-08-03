"""
Wrapper de inferencia para la categoría de super-resolución/BWE.

Variante activa: AudioSR (paquete `audiosr`, haoheliu/versatile_audio_super_resolution).

Lógica de chunking + crossfade calcada de la ya validada en la notebook
05_AudioSR.ipynb: chunks de 5s (CHUNK_SECONDS), solape de 0.1s
(OVERLAP_SECONDS) para crossfade lineal, umbral de silencio por RMS
(SILENCE_RMS_THRESHOLD) para saltarse AudioSR en chunks silenciosos, y
fallback a `librosa.resample` clásico tanto en chunks silenciosos como si
AudioSR lanza ValueError en un chunk concreto (p.ej. el bug conocido de
`scipy.iirfilter: Wn must be 0 < Wn < 1` en roll-off espectral degenerado).

Instalación (recordatorio para la celda de la notebook de la app):
audiosr==0.0.7 declara numpy<=1.23.5, que no compila en Python 3.12 (Colab
actual) -> se instala con --no-deps y se añaden a mano las dependencias
transitorias que sí hacen falta (unidecode, phonemizer, ftfy, torchlibrosa).

Limitación arquitectónica real (no configurable): AudioSR solo admite clips
de hasta ~5.12s -> con audio más largo sin trocear, CUDA OOM.

AudioSR devuelve SIEMPRE la salida en mono y a 48kHz fijo, independientemente
del sample rate/canales de entrada -> se fuerza mono desde la carga inicial
(igual que en la notebook).

Diferencia respecto a la notebook: allí el resultado final se guardaba
directamente como int16 (audio * 32767) sin comprobar el pico antes. Aquí se
añade la misma normalización de pico (normalizar_pico, pico_objetivo=0.95)
que usan el resto de wrappers de la app, por si AudioSR genera algún pico
>1.0 en el bloque empírico (mismo tipo de caso ya visto con DeepFilterNet3).

Variante pendiente: MossFormer2 (combinado) — último modelo por conectar.
"""

import os
import tempfile

import numpy as np
import torch

from tfg_models import model_manager

CHUNK_SECONDS = 5.0      # margen bajo el límite de 5.12s recomendado
OVERLAP_SECONDS = 0.1    # solape pequeño para crossfade y evitar clicks
SR_SALIDA = 48000
SILENCE_RMS_THRESHOLD = 1e-4  # ajustable si se detectan falsos positivos/negativos


def _cargar_audiosr():
    from audiosr import build_model

    device = "cuda" if torch.cuda.is_available() else "cpu"
    modelo = build_model(model_name="basic", device=device)
    return {"modelo": modelo, "device": device}


def procesar(ruta_audio: str, ruta_salida: str):
    """
    Ejecuta AudioSR (BWE) sobre un archivo de audio en disco, troceando en
    chunks de 5s con crossfade (misma lógica que 05_AudioSR.ipynb), y guarda
    el resultado en ruta_salida.

    Returns:
        audio_mejorado_arr (np.ndarray): señal de salida mono a 48kHz, ya
            normalizada (pico 0.95).
        sr_modelo (int): 48000 Hz, fijo por AudioSR.
        pico_antes (float): pico máximo absoluto ANTES de normalizar.
    """
    import librosa
    import soundfile as sf
    from audiosr import super_resolution

    from tfg_audio_utils.audio_utils_funcionescomunes import normalizar_pico, guardar_audio

    paquete = model_manager.obtener_modelo("bwe", "audiosr", _cargar_audiosr)
    modelo_audiosr = paquete["modelo"]

    # Carga forzada a mono, con forma (1, muestras) -- igual que en la notebook,
    # para mantener el mismo pipeline entre chunks procesados por el modelo y
    # chunks silenciosos resampleados aparte.
    audio, sr_in = librosa.load(ruta_audio, sr=None, mono=True)
    audio = audio[np.newaxis, :]

    n_channels, n_samples = audio.shape
    chunk_len_in = int(CHUNK_SECONDS * sr_in)
    overlap_len_in = int(OVERLAP_SECONDS * sr_in)

    starts = list(range(0, n_samples, chunk_len_in - overlap_len_in))

    chunks_out = []

    with tempfile.TemporaryDirectory() as tmpdir:
        for i, start in enumerate(starts):
            end = min(start + chunk_len_in, n_samples)
            chunk_audio = audio[:, start:end]

            rms = np.sqrt(np.mean(chunk_audio**2))

            if rms < SILENCE_RMS_THRESHOLD:
                # Chunk silencioso: se resamplea sin pasar por AudioSR.
                chunk_out = librosa.resample(chunk_audio, orig_sr=sr_in, target_sr=SR_SALIDA)
                chunks_out.append(chunk_out)
                continue

            chunk_path = os.path.join(tmpdir, f"chunk_{i}.wav")
            sf.write(chunk_path, chunk_audio.T, sr_in)

            try:
                waveform = super_resolution(
                    modelo_audiosr,
                    chunk_path,
                    seed=42,
                    guidance_scale=3.5,
                    ddim_steps=50,
                    latent_t_per_second=12.8,
                )
                chunk_out = waveform[0]  # (1, muestras) a 48kHz
            except ValueError:
                # Fallo conocido de AudioSR en chunks con roll-off espectral
                # degenerado (scipy.iirfilter: "Wn must be 0 < Wn < 1").
                chunk_out = librosa.resample(chunk_audio, orig_sr=sr_in, target_sr=SR_SALIDA)

            if chunk_out.ndim == 1:
                chunk_out = chunk_out[np.newaxis, :]

            chunks_out.append(chunk_out)
            torch.cuda.empty_cache()

    # Recomposición con crossfade lineal en la zona de solape (ya a 48kHz)
    overlap_len_out = int(OVERLAP_SECONDS * SR_SALIDA)
    final_audio = chunks_out[0]

    for chunk in chunks_out[1:]:
        cabe_solape = (
            overlap_len_out > 0
            and final_audio.shape[1] >= overlap_len_out
            and chunk.shape[1] >= overlap_len_out
        )
        if cabe_solape:
            fade_out = np.linspace(1, 0, overlap_len_out)
            fade_in = np.linspace(0, 1, overlap_len_out)

            final_tail = final_audio[:, -overlap_len_out:] * fade_out
            chunk_head = chunk[:, :overlap_len_out] * fade_in
            crossfaded = final_tail + chunk_head

            final_audio = np.concatenate(
                [final_audio[:, :-overlap_len_out], crossfaded, chunk[:, overlap_len_out:]],
                axis=1,
            )
        else:
            final_audio = np.concatenate([final_audio, chunk], axis=1)

    audio_mejorado_arr = final_audio.squeeze()

    pico_antes = np.max(np.abs(audio_mejorado_arr))
    audio_mejorado_arr = normalizar_pico(audio_mejorado_arr, pico_objetivo=0.95)

    guardar_audio(ruta_salida, audio_mejorado_arr, SR_SALIDA)

    return audio_mejorado_arr, SR_SALIDA, pico_antes
