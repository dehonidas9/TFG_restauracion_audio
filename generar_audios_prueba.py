"""
Generador de audios sintéticos degradados para el bloque empírico del TFG.

VERSION AUTOCONTENIDA: solo necesita saber dónde está la carpeta raíz del
TFG (BASE_DIR) y los 3 audios de entrada. Crea su propia carpeta de salida
nueva (no toca ninguna carpeta que ya tengas) y valida que todo exista
ANTES de procesar nada, imprimiendo un mensaje claro si falta algo, en vez
de un traceback a mitad de ejecución.

A partir de dos fragmentos limpios de 10s (grabación de estudio, narración
de las primeras páginas del Quijote) y un audio de interferencia (no vocal,
para la categoría de separación), genera audios degradados en 5 categorías
con un barrido de severidad controlado, y escribe un manifiesto CSV con los
parámetros de cada archivo generado.

FRAGMENTO 1: barrido completo (todos los niveles).
FRAGMENTO 2: solo los 2 niveles marcados como "robustez" por categoría.
"""

import csv
import os
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import butter, fftconvolve, sosfilt

# ---------------------------------------------------------------------------
# CONFIGURACIÓN — solo hay que tocar esto
# ---------------------------------------------------------------------------

# Carpeta raíz del TFG. Todo lo demás se calcula a partir de aquí.
BASE_DIR = Path(r"C:\Users\User\Desktop\UNIVERSIDAD\quinto de carrera\TFG TELECO")

RUTA_FRAGMENTO_1 = BASE_DIR / "audio_prueba" / "audio1.wav"
RUTA_FRAGMENTO_2 = BASE_DIR / "audio_prueba" / "audio2.wav"
RUTA_INTERFERENCIA = BASE_DIR / "audio_prueba" / "The Ciggie (Original Mix).mp3"

# Carpeta de salida NUEVA — el script la crea ella sola, no escribe en
# ninguna carpeta que ya tuvieras de antes.
OUTPUT_DIR = BASE_DIR / "audios_sinteticos_barrido"
RUTA_MANIFIESTO = OUTPUT_DIR / "manifiesto_audios_prueba.csv"

SEMILLA = 42  # reproducibilidad (ruido, RIR aleatoria si aplica)

# ---------------------------------------------------------------------------
# Barrido de severidad por categoría
# ---------------------------------------------------------------------------

NIVELES = {
    "denoising": [-5, 0, 5, 10, 15, 20],
    "dereverberation": [0.3, 0.6, 0.9, 1.2, 1.6, 2.0],
    "bwe": [1000, 2000, 3000, 4000, 6000],
    "declipping": [5, 10, 20, 30, 40, 50],
    "separacion": [-10, -5, 0, 5, 10, 15],
}

NIVELES_ROBUSTEZ = {
    "denoising": (5, -5),
    "dereverberation": (0.9, 2.0),
    "bwe": (3000, 1000),
    "declipping": (20, 50),
    "separacion": (0, -10),
}

NOMBRE_PARAMETRO = {
    "denoising": "snr_db",
    "dereverberation": "t60_s",
    "bwe": "cutoff_hz",
    "declipping": "pct_saturado",
    "separacion": "sir_db",
}


# ---------------------------------------------------------------------------
# Validación previa (falla rápido, con mensajes claros)
# ---------------------------------------------------------------------------

def validar_entradas():
    faltan = []
    for nombre, ruta in [
        ("RUTA_FRAGMENTO_1", RUTA_FRAGMENTO_1),
        ("RUTA_FRAGMENTO_2", RUTA_FRAGMENTO_2),
        ("RUTA_INTERFERENCIA", RUTA_INTERFERENCIA),
    ]:
        existe = ruta.exists()
        print(f"  {nombre}: {ruta}  {'OK' if existe else '-> NO ENCONTRADO'}")
        if not existe:
            faltan.append(nombre)

    if faltan:
        print()
        print("No puedo seguir: faltan los archivos de arriba marcados como NO ENCONTRADO.")
        print("Revisa el nombre exacto y la ruta en el bloque CONFIGURACIÓN del script.")
        sys.exit(1)

    print("Todas las entradas encontradas correctamente.\n")


# ---------------------------------------------------------------------------
# Carga de audio (librosa: soporta wav y mp3 sin líos de backend)
# ---------------------------------------------------------------------------

def _cargar_audio(ruta):
    import librosa

    audio, sr = librosa.load(str(ruta), sr=None, mono=True)
    return audio.astype(np.float32), sr


def _recortar_a_10s_y_guardar(ruta, duracion_s=10.0):
    """
    Recorta el archivo en `ruta` a sus primeros `duracion_s` segundos (donde
    está el habla real) y SOBRESCRIBE el propio archivo en el sitio donde ya
    estaba, para que quede así de forma permanente y no haya que repetir
    este recorte cada vez que se corra el script.
    """
    audio, sr = _cargar_audio(ruta)
    n_muestras = int(duracion_s * sr)
    audio_recortado = audio[:n_muestras]
    sf.write(str(ruta), audio_recortado, sr)
    return audio_recortado, sr


def _guardar_audio(ruta, audio, sr):
    ruta.parent.mkdir(parents=True, exist_ok=True)
    pico = np.max(np.abs(audio))
    if pico > 0.99:
        audio = audio / pico * 0.95
    sf.write(str(ruta), audio, sr)


# ---------------------------------------------------------------------------
# Funciones de degradación
# ---------------------------------------------------------------------------

def degradar_denoising(audio, sr, snr_db, rng):
    ruido = rng.normal(0, 1, size=audio.shape).astype(np.float32)
    potencia_audio = np.mean(audio**2)
    potencia_ruido = np.mean(ruido**2)
    factor = np.sqrt(potencia_audio / (potencia_ruido * (10 ** (snr_db / 10))))
    return audio + ruido * factor


def degradar_dereverb(audio, sr, t60, rng):
    """
    RIR sintética mediante ruido gaussiano con envolvente de decaimiento
    exponencial (modelo estocástico de Polack) -- controla el T60 con una
    relación cerrada, sin simular geometría de sala. Mucho más rápido y
    predecible que el método de fuentes-imagen (pyroomacoustics), que puede
    tardar minutos por RIR para T60 altos según la sala/orden de reflexión.

    T60 = tiempo para que el nivel caiga 60dB => tau = T60 / (3*ln(10)).
    """
    duracion_rir = t60 + 0.3  # cola un poco más larga que el propio T60
    n_muestras_rir = max(int(sr * duracion_rir), 1)
    tau = t60 / (3 * np.log(10))

    t = np.arange(n_muestras_rir) / sr
    envolvente = np.exp(-t / tau)
    rir = rng.normal(0, 1, size=n_muestras_rir).astype(np.float32) * envolvente
    rir = rir / np.sqrt(np.sum(rir**2))  # normaliza energía de la RIR a 1

    salida = fftconvolve(audio, rir, mode="full")[: len(audio) + int(sr * 0.3)]
    if len(salida) < len(audio):
        salida = np.pad(salida, (0, len(audio) - len(salida)))
    return salida.astype(np.float32)


def degradar_bwe(audio, sr, cutoff_hz):
    sos = butter(8, cutoff_hz, btype="low", fs=sr, output="sos")
    return sosfilt(sos, audio).astype(np.float32)


def degradar_declipping(audio, pct_saturado):
    percentil_objetivo = 100 - pct_saturado
    umbral = np.percentile(np.abs(audio), percentil_objetivo)
    umbral = max(umbral, 1e-6)
    return np.clip(audio, -umbral, umbral).astype(np.float32)


def generar_mezcla_separacion(audio_objetivo, audio_interferencia, sir_db, rng):
    n = len(audio_objetivo)
    if len(audio_interferencia) < n:
        reps = int(np.ceil(n / len(audio_interferencia)))
        audio_interferencia = np.tile(audio_interferencia, reps)
    inicio = (
        rng.integers(0, len(audio_interferencia) - n + 1)
        if len(audio_interferencia) > n
        else 0
    )
    interferencia = audio_interferencia[inicio : inicio + n]

    potencia_objetivo = np.mean(audio_objetivo**2)
    potencia_interferencia = np.mean(interferencia**2)
    factor = np.sqrt(potencia_objetivo / (potencia_interferencia * (10 ** (sir_db / 10))))

    return audio_objetivo + interferencia * factor


# ---------------------------------------------------------------------------
# Orquestación
# ---------------------------------------------------------------------------

def procesar_categoria(categoria, audio_limpio, sr, fragmento_id, niveles, rng,
                        ruta_limpio_referencia, audio_interferencia=None):
    filas = []
    parametro = NOMBRE_PARAMETRO[categoria]

    for nivel in niveles:
        print(f"  [{categoria}] fragmento {fragmento_id}, {parametro}={nivel} ...")

        if categoria == "denoising":
            degradado = degradar_denoising(audio_limpio, sr, nivel, rng)
        elif categoria == "dereverberation":
            degradado = degradar_dereverb(audio_limpio, sr, nivel, rng)
        elif categoria == "bwe":
            degradado = degradar_bwe(audio_limpio, sr, nivel)
        elif categoria == "declipping":
            degradado = degradar_declipping(audio_limpio, nivel)
        elif categoria == "separacion":
            degradado = generar_mezcla_separacion(audio_limpio, audio_interferencia, nivel, rng)
        else:
            raise ValueError(f"Categoría desconocida: {categoria}")

        nombre_archivo = f"{categoria}_frag{fragmento_id}_{parametro}_{nivel}.wav"
        ruta_salida = OUTPUT_DIR / categoria / nombre_archivo
        _guardar_audio(ruta_salida, degradado, sr)

        filas.append(
            {
                "categoria": categoria,
                "fragmento": fragmento_id,
                "parametro_severidad": parametro,
                "valor_severidad": nivel,
                "ruta_audio_limpio": str(ruta_limpio_referencia),
                "ruta_audio_degradado": str(ruta_salida),
            }
        )

    return filas


def main():
    print("Comprobando archivos de entrada:")
    validar_entradas()

    rng = np.random.default_rng(SEMILLA)

    print("Recortando los fragmentos a los primeros 10 segundos (habla real) "
          "y sobrescribiéndolos en su carpeta original...")
    audio_f1, sr1 = _recortar_a_10s_y_guardar(RUTA_FRAGMENTO_1)
    print(f"  {RUTA_FRAGMENTO_1.name}: recortado y guardado ({len(audio_f1) / sr1:.2f}s)")
    audio_f2, sr2 = _recortar_a_10s_y_guardar(RUTA_FRAGMENTO_2)
    print(f"  {RUTA_FRAGMENTO_2.name}: recortado y guardado ({len(audio_f2) / sr2:.2f}s)")

    print("\nCargando audio de interferencia...")
    audio_interferencia, sr_interf = _recortar_a_10s_y_guardar(RUTA_INTERFERENCIA)

    sr = sr1
    if sr2 != sr:
        import librosa
        audio_f2 = librosa.resample(audio_f2, orig_sr=sr2, target_sr=sr)
    if sr_interf != sr:
        import librosa
        audio_interferencia = librosa.resample(audio_interferencia, orig_sr=sr_interf, target_sr=sr)

    print(f"Sample rate de trabajo: {sr} Hz\n")

    todas_las_filas = []

    print("Generando barrido completo (fragmento 1)...")
    for categoria, niveles in NIVELES.items():
        todas_las_filas += procesar_categoria(
            categoria, audio_f1, sr, fragmento_id=1, niveles=niveles, rng=rng,
            ruta_limpio_referencia=RUTA_FRAGMENTO_1,
            audio_interferencia=audio_interferencia,
        )

    print("\nGenerando verificación de robustez (fragmento 2)...")
    for categoria, niveles_robustez in NIVELES_ROBUSTEZ.items():
        todas_las_filas += procesar_categoria(
            categoria, audio_f2, sr, fragmento_id=2, niveles=list(niveles_robustez), rng=rng,
            ruta_limpio_referencia=RUTA_FRAGMENTO_2,
            audio_interferencia=audio_interferencia,
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(RUTA_MANIFIESTO, "w", newline="") as f:
        campos = ["categoria", "fragmento", "parametro_severidad", "valor_severidad",
                  "ruta_audio_limpio", "ruta_audio_degradado"]
        writer = csv.DictWriter(f, fieldnames=campos)
        writer.writeheader()
        writer.writerows(todas_las_filas)

    print(f"\nListo. Generados {len(todas_las_filas)} audios degradados.")
    print(f"Carpeta de salida: {OUTPUT_DIR}")
    print(f"Manifiesto: {RUTA_MANIFIESTO}")


if __name__ == "__main__":
    main()
