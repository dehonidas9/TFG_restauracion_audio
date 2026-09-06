"""
Calcula métricas (intrusivas + DNSMOS) para el barrido de audios sintéticos,
pasando cada audio degradado por los modelos de IA correspondientes a su
categoría, SIN pasar por la interfaz de Gradio. Escribe un único CSV con
una fila por (condición, variante) — original_degradado + cada modelo de IA.

Requisitos antes de correr (dentro del venv311, en Colab):
    !/content/env311/bin/python -m pip install -q pesq pystoi

Uso: correr desde dentro de Colab, con la GPU y el resto del stack de la
app ya disponibles (mismo entorno que usa launch_app_venv.py).

Entradas esperadas:
  - La carpeta `audios_sinteticos_barrido` (generada localmente con
    generar_audios_prueba.py) subida a Drive, con sus subcarpetas por
    categoría y el manifiesto_audios_prueba.csv dentro.
  - Los dos fragmentos limpios (audio1.wav, audio2.wav) también subidos a
    Drive, en la carpeta indicada en DRIVE_FRAGMENTOS_DIR.

El script IGNORA las rutas absolutas de Windows que trae el CSV del
manifiesto (no existen en Colab) y reconstruye las rutas reales a partir
del nombre de archivo + las carpetas DRIVE_* de abajo.
"""

import csv
import ntpath
import os
import sys
import tempfile

import numpy as np

# ---------------------------------------------------------------------------
# CONFIGURACIÓN
# ---------------------------------------------------------------------------

PROJECT_ROOT = "/content/drive/MyDrive/Proyecto_Audio"

# ---------------------------------------------------------------------------
# Parches de compatibilidad -- LOS MISMOS que ya usa launch_app_venv.py.
# Sin esto, torchaudio.load() intenta usar torchcodec (no instalado) y
# torch.load() falla por el default weights_only=True de PyTorch 2.6+.
# Deben aplicarse ANTES de importar tfg_models (que internamente usa
# torch/torchaudio/huggingface_hub).
# ---------------------------------------------------------------------------

import huggingface_hub


def _compatibilizar_hf_hub_download(func_original):
    def wrapper(*args, **kwargs):
        if "use_auth_token" in kwargs:
            kwargs.setdefault("token", kwargs.pop("use_auth_token"))
        return func_original(*args, **kwargs)
    return wrapper


huggingface_hub.hf_hub_download = _compatibilizar_hf_hub_download(huggingface_hub.hf_hub_download)
huggingface_hub.file_download.hf_hub_download = huggingface_hub.hf_hub_download
if hasattr(huggingface_hub, "snapshot_download"):
    huggingface_hub.snapshot_download = _compatibilizar_hf_hub_download(huggingface_hub.snapshot_download)

import torch
import torchaudio
import soundfile as _sf_patch

_torch_load_original = torch.load


def _torch_load_compat(*args, **kwargs):
    kwargs["weights_only"] = False
    return _torch_load_original(*args, **kwargs)


torch.load = _torch_load_compat


def _torchaudio_load_via_soundfile(filepath, *args, **kwargs):
    data, sr = _sf_patch.read(filepath, always_2d=True)
    waveform = torch.from_numpy(data.T).float()
    return waveform, sr


torchaudio.load = _torchaudio_load_via_soundfile

import types

os.environ["MPLBACKEND"] = "Agg"
os.environ["PATH"] = f"/content/env311/bin:{os.environ['PATH']}"

if not hasattr(torchaudio, "backend"):
    backend_mod = types.ModuleType("torchaudio.backend")
    common_mod = types.ModuleType("torchaudio.backend.common")

    class AudioMetaData:
        def __init__(self, sample_rate=0, num_frames=0, num_channels=0, bits_per_sample=0, encoding=""):
            self.sample_rate = sample_rate
            self.num_frames = num_frames
            self.num_channels = num_channels
            self.bits_per_sample = bits_per_sample
            self.encoding = encoding

    common_mod.AudioMetaData = AudioMetaData
    backend_mod.common = common_mod
    sys.modules["torchaudio.backend"] = backend_mod
    sys.modules["torchaudio.backend.common"] = common_mod
    torchaudio.backend = backend_mod

os.environ["HF_HOME"] = f"{PROJECT_ROOT}/cache"
os.environ["HF_HUB_CACHE"] = f"{PROJECT_ROOT}/cache"

# ---------------------------------------------------------------------------

sys.path.insert(0, f"{PROJECT_ROOT}/gradio_app")

DRIVE_MANIFIESTO = f"{PROJECT_ROOT}/audios_sinteticos_barrido/manifiesto_audios_prueba.csv"
DRIVE_AUDIOS_DIR = f"{PROJECT_ROOT}/audios_sinteticos_barrido"  # contiene las subcarpetas por categoría
DRIVE_FRAGMENTOS_DIR = f"{PROJECT_ROOT}/audio_prueba"  # contiene audio1.wav y audio2.wav

RUTA_CSV_SALIDA = f"{PROJECT_ROOT}/audios_sinteticos_barrido/metricas_resultado.csv"

FRAGMENTO_A_NOMBRE = {1: "audio1.wav", 2: "audio2.wav"}

# categoría (tal como aparece en el manifiesto) -> [(nombre_variante, función_inferencia)]
CATEGORIA_A_VARIANTES = {}  # se rellena en main() tras importar tfg_models

# categoría -> qué métricas calcular (evita gastar tiempo en métricas que no aplican)
METRICAS_POR_CATEGORIA = {
    "denoising": {"pesq", "stoi", "sisdr", "dnsmos"},
    "dereverberation": {"pesq", "stoi", "sisdr", "dnsmos"},
    "bwe": {"lsd", "stoi", "dnsmos"},
    "declipping": {"pesq", "sisdr", "dnsmos"},
    "separacion": {"sisdr", "dnsmos"},
}


# ---------------------------------------------------------------------------
# Métricas
# ---------------------------------------------------------------------------

def _alinear_longitud(a, b):
    n = min(len(a), len(b))
    return a[:n], b[:n]


def calcular_pesq(ref, sr_ref, deg, sr_deg):
    from pesq import pesq
    import librosa

    ref16 = librosa.resample(ref, orig_sr=sr_ref, target_sr=16000) if sr_ref != 16000 else ref
    deg16 = librosa.resample(deg, orig_sr=sr_deg, target_sr=16000) if sr_deg != 16000 else deg
    ref16, deg16 = _alinear_longitud(ref16, deg16)
    try:
        return pesq(16000, ref16, deg16, "wb")
    except Exception as e:
        print(f"    [aviso] PESQ falló: {e}")
        return None


def calcular_stoi(ref, sr_ref, deg, sr_deg):
    from pystoi import stoi
    import librosa

    if sr_deg != sr_ref:
        deg = librosa.resample(deg, orig_sr=sr_deg, target_sr=sr_ref)
    ref_a, deg_a = _alinear_longitud(ref, deg)
    try:
        return stoi(ref_a, deg_a, sr_ref, extended=False)
    except Exception as e:
        print(f"    [aviso] STOI falló: {e}")
        return None


def calcular_sisdr(ref, sr_ref, deg, sr_deg):
    import librosa

    if sr_deg != sr_ref:
        deg = librosa.resample(deg, orig_sr=sr_deg, target_sr=sr_ref)
    ref_a, deg_a = _alinear_longitud(ref, deg)
    ref_a = ref_a - np.mean(ref_a)
    deg_a = deg_a - np.mean(deg_a)
    alpha = np.dot(deg_a, ref_a) / (np.dot(ref_a, ref_a) + 1e-10)
    proyeccion = alpha * ref_a
    ruido = deg_a - proyeccion
    return float(10 * np.log10(np.sum(proyeccion**2) / (np.sum(ruido**2) + 1e-10)))


def calcular_lsd(ref, sr_ref, deg, sr_deg, n_fft=1024, hop=256):
    import librosa
    from scipy.signal import stft

    if sr_deg != sr_ref:
        deg = librosa.resample(deg, orig_sr=sr_deg, target_sr=sr_ref)
    ref_a, deg_a = _alinear_longitud(ref, deg)

    _, _, R = stft(ref_a, fs=sr_ref, nperseg=n_fft, noverlap=n_fft - hop)
    _, _, D = stft(deg_a, fs=sr_ref, nperseg=n_fft, noverlap=n_fft - hop)
    n = min(R.shape[1], D.shape[1])
    R, D = R[:, :n], D[:, :n]

    diferencia = (20 * np.log10(np.abs(R) + 1e-8) - 20 * np.log10(np.abs(D) + 1e-8)) ** 2
    return float(np.mean(np.sqrt(np.mean(diferencia, axis=0))))


def calcular_todas_las_metricas(ref, sr_ref, deg, sr_deg, metricas_necesarias):
    from tfg_audio_utils.audio_utils_metricas_no_intrusivas import calcular_dnsmos

    resultado = {"pesq": None, "stoi": None, "sisdr": None, "lsd": None,
                 "dnsmos_ovrl": None, "dnsmos_sig": None, "dnsmos_bak": None}

    if "pesq" in metricas_necesarias:
        resultado["pesq"] = calcular_pesq(ref, sr_ref, deg, sr_deg)
    if "stoi" in metricas_necesarias:
        resultado["stoi"] = calcular_stoi(ref, sr_ref, deg, sr_deg)
    if "sisdr" in metricas_necesarias:
        resultado["sisdr"] = calcular_sisdr(ref, sr_ref, deg, sr_deg)
    if "lsd" in metricas_necesarias:
        resultado["lsd"] = calcular_lsd(ref, sr_ref, deg, sr_deg)
    if "dnsmos" in metricas_necesarias:
        dnsmos = calcular_dnsmos(deg, sr_deg)
        resultado["dnsmos_ovrl"] = dnsmos.get("ovrl_mos")
        resultado["dnsmos_sig"] = dnsmos.get("sig_mos")
        resultado["dnsmos_bak"] = dnsmos.get("bak_mos")

    return resultado


# ---------------------------------------------------------------------------
# Orquestación
# ---------------------------------------------------------------------------

def _cargar_audio(ruta):
    import librosa

    audio, sr = librosa.load(ruta, sr=None, mono=True)
    return audio.astype(np.float32), sr


def main():
    import librosa  # noqa: F401 (fuerza el import temprano, falla rápido si falta)
    from tfg_models import denoising, dereverb, declipping, separacion_fuentes, mossformer2
    from tfg_models import bwe as bwe_mod

    CATEGORIA_A_VARIANTES.update({
        "denoising": [("deepfilternet", denoising.procesar), ("mossformer2", mossformer2.procesar)],
        "dereverberation": [("sgmse", dereverb.procesar), ("mossformer2", mossformer2.procesar)],
        "bwe": [("audiosr", bwe_mod.procesar), ("mossformer2", mossformer2.procesar)],
        "declipping": [("voicefixer", declipping.procesar)],
        "separacion": [("htdemucs", separacion_fuentes.procesar)],
    })

    with open(DRIVE_MANIFIESTO, newline="") as f:
        filas_manifiesto = list(csv.DictReader(f))

    print(f"{len(filas_manifiesto)} condiciones en el manifiesto.\n")

    # Cache de audios limpios (solo 2 fragmentos, se leen una vez)
    cache_limpios = {}

    filas_resultado = []

    with tempfile.TemporaryDirectory() as tmpdir:
        for i, fila in enumerate(filas_manifiesto):
            categoria = fila["categoria"]
            fragmento = int(fila["fragmento"])
            parametro = fila["parametro_severidad"]
            valor = fila["valor_severidad"]

            # ntpath.basename (no os.path.basename): el manifiesto trae rutas
            # de Windows ("\") generadas en local, pero esto corre en Linux/
            # Colab, donde os.path.basename solo entiende "/" y devolvería
            # la ruta de Windows entera como si fuera el nombre de archivo.
            nombre_degradado = ntpath.basename(fila["ruta_audio_degradado"])
            ruta_degradado = os.path.join(DRIVE_AUDIOS_DIR, categoria, nombre_degradado)

            if fragmento not in cache_limpios:
                ruta_limpio = os.path.join(DRIVE_FRAGMENTOS_DIR, FRAGMENTO_A_NOMBRE[fragmento])
                cache_limpios[fragmento] = _cargar_audio(ruta_limpio)
            audio_limpio, sr_limpio = cache_limpios[fragmento]

            metricas_necesarias = METRICAS_POR_CATEGORIA[categoria]

            print(f"[{i+1}/{len(filas_manifiesto)}] {categoria} frag{fragmento} {parametro}={valor}")

            # --- Fila del audio degradado sin procesar ---
            audio_degradado, sr_degradado = _cargar_audio(ruta_degradado)
            metricas_original = calcular_todas_las_metricas(
                audio_limpio, sr_limpio, audio_degradado, sr_degradado, metricas_necesarias
            )
            filas_resultado.append({
                "categoria": categoria, "fragmento": fragmento, "variante": "original_degradado",
                "parametro_severidad": parametro, "valor_severidad": valor,
                "ruta_audio": ruta_degradado, **metricas_original,
            })

            # --- Una fila por cada variante de IA aplicable a esta categoría ---
            for nombre_variante, funcion_inferencia in CATEGORIA_A_VARIANTES[categoria]:
                ruta_salida_ia = os.path.join(tmpdir, f"salida_{nombre_variante}.wav")
                try:
                    audio_ia, sr_ia, _ = funcion_inferencia(ruta_degradado, ruta_salida_ia)
                except Exception as e:
                    print(f"    [ERROR] {nombre_variante} falló en este audio: {e}")
                    continue

                metricas_ia = calcular_todas_las_metricas(
                    audio_limpio, sr_limpio, audio_ia, sr_ia, metricas_necesarias
                )
                filas_resultado.append({
                    "categoria": categoria, "fragmento": fragmento, "variante": nombre_variante,
                    "parametro_severidad": parametro, "valor_severidad": valor,
                    "ruta_audio": ruta_salida_ia, **metricas_ia,
                })

            # Guardado incremental: si algo se corta a mitad (Colab, GPU, etc.)
            # no se pierde lo ya calculado.
            with open(RUTA_CSV_SALIDA, "w", newline="") as f:
                campos = ["categoria", "fragmento", "variante", "parametro_severidad",
                          "valor_severidad", "ruta_audio", "pesq", "stoi", "sisdr", "lsd",
                          "dnsmos_ovrl", "dnsmos_sig", "dnsmos_bak"]
                writer = csv.DictWriter(f, fieldnames=campos)
                writer.writeheader()
                writer.writerows(filas_resultado)

    print(f"\nListo. {len(filas_resultado)} filas escritas en {RUTA_CSV_SALIDA}")


if __name__ == "__main__":
    main()
