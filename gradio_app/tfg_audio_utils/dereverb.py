"""
Wrapper de inferencia para la categoría de dereverberation.

Variante activa: SGMSE+ (individual), vía el paquete `sgmse`
(github.com/sp-uhh/sgmse), modelo generativo basado en difusión (score-based)
con checkpoint específico de dereverberación (WSJ0-REVERB).

Sustituye a MP-SENet como variante activa: el paquete de inferencia de
MP-SENet (JacobLinCool/MPSENet) solo expone checkpoints de denoising
(VoiceBank+DEMAND y DNS-Challenge). No existe públicamente ningún checkpoint
de MP-SENet entrenado para dereverberación, ni en el repositorio original
(yxlu-0102/MP-SENet) ni en su Hugging Face — confirmado al buscar
específicamente esta variante. El wrapper de MP-SENet se conserva en
mpsenet_dereverb.py como caso de estudio de desajuste bibliográfico-empírico
(apartado 4.4.2 de la memoria).

Nota de rendimiento: al ser un proceso de difusión con muestreo iterativo
(no una pasada feed-forward), la inferencia es notablemente más lenta que la
del resto de modelos de la app — del orden de decenas de segundos por audio
en una T4, con N=50 pasos de muestreo (parámetros recomendados por los
autores para este checkpoint: N=50, snr=0.33, sampler PC, corrector "ald").

Requiere, antes de importar este módulo (celda de setup en Colab, ver README):

    !git clone https://github.com/sp-uhh/sgmse.git /content/sgmse
    !pip install -e /content/sgmse -q
    !pip install gdown -q
    !gdown 1eiOy0VjHh9V9ZUFTxu1Pq2w19izl9ejD -O {ruta_del_checkpoint}

Dónde se guarda el checkpoint es configurable (ver CHECKPOINT_SGMSE más abajo):
por defecto se busca en una carpeta `pesos/` junto a la raíz del proyecto
(donde vive `app.py`), para no atar el código a la cuenta de Drive de un
usuario concreto. Puede sobreescribirse con la variable de entorno
SGMSE_CHECKPOINT si se prefiere guardarlo en otra ubicación.
"""

import os

import numpy as np
import torch
import librosa

from tfg_models import model_manager

# Ruta del checkpoint de SGMSE+ (WSJ0-REVERB). Por defecto, "pesos/" en la
# raíz del proyecto (hermana de tfg_models/ y de app.py); configurable con
# la variable de entorno SGMSE_CHECKPOINT para no depender de una ruta de
# Drive concreta.
_RAIZ_PROYECTO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHECKPOINT_SGMSE = os.environ.get(
    "SGMSE_CHECKPOINT",
    os.path.join(_RAIZ_PROYECTO, "pesos", "sgmse_wsj0_reverb.ckpt"),
)

# Parámetros de muestreo recomendados por los autores para el checkpoint
# WSJ0-REVERB (ver README de sp-uhh/sgmse).
SGMSE_N = 50
SGMSE_SNR = 0.33
SGMSE_CORRECTOR = "ald"
SGMSE_CORRECTOR_STEPS = 1
SGMSE_T_EPS = 0.03


def _cargar_sgmse():
    from sgmse.model import ScoreModel

    if not os.path.isfile(CHECKPOINT_SGMSE):
        raise FileNotFoundError(
            f"No se encuentra el checkpoint de SGMSE+ en '{CHECKPOINT_SGMSE}'. "
            "Descárgalo siguiendo las instrucciones del README (celda de "
            "instalación de SGMSE+) o define la variable de entorno "
            "SGMSE_CHECKPOINT apuntando a su ubicación."
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    modelo = ScoreModel.load_from_checkpoint(CHECKPOINT_SGMSE, map_location=device)
    modelo.t_eps = SGMSE_T_EPS
    modelo.eval()
    return {"modelo": modelo, "device": device}


def procesar(ruta_audio: str, ruta_salida: str):
    """
    Ejecuta inferencia de SGMSE+ sobre un archivo de audio en disco y guarda
    el resultado en ruta_salida.

    Replica la lógica de enhancement.py del repositorio original
    (STFT -> muestreo inverso por difusión -> vuelta al dominio temporal),
    reescrita aquí para operar en memoria sobre un único archivo en vez de
    por CLI sobre un directorio, e integrada con model_manager para reusar
    el modelo cargado entre llamadas (mismo patrón que el resto de wrappers
    de esta app).

    Returns:
        audio_mejorado_arr (np.ndarray): señal de salida ya normalizada (pico 0.95).
        sr_modelo (int): sample rate propio del checkpoint (16 kHz para
            WSJ0-REVERB / backbone ncsnpp estándar).
        pico_antes (float): pico máximo absoluto ANTES de normalizar. Se
            mantiene el mismo criterio que en el resto de wrappers, útil
            para detectar desbordamientos en el bloque empírico.
    """
    from tfg_audio_utils.audio_utils_funcionescomunes import normalizar_pico, guardar_audio
    from sgmse.util.other import pad_spec

    paquete = model_manager.obtener_modelo("dereverberation", "sgmse", _cargar_sgmse)
    modelo = paquete["modelo"]
    device = paquete["device"]

    # Sample rate objetivo según el backbone del checkpoint (igual que en
    # enhancement.py); WSJ0-REVERB usa 16 kHz con padding "zero_pad".
    if modelo.backbone == "ncsnpp_48k":
        sr_modelo, pad_mode = 48000, "reflection"
    elif modelo.backbone == "ncsnpp_v2":
        sr_modelo, pad_mode = 16000, "reflection"
    else:
        sr_modelo, pad_mode = 16000, "zero_pad"

    # librosa.load resamplea directamente al sr propio del checkpoint
    audio_entrada, _ = librosa.load(ruta_audio, sr=sr_modelo)
    y = torch.tensor(audio_entrada).unsqueeze(0)

    T_orig = y.size(1)

    # Normalización previa al modelo (requerida por SGMSE, distinta de la
    # normalizar_pico de salida que se aplica al final).
    norm_factor = y.abs().max()
    y = y / norm_factor

    # Preparación de la entrada de la red (STFT + padding espectral)
    Y = torch.unsqueeze(modelo._forward_transform(modelo._stft(y.to(device))), 0)
    Y = pad_spec(Y, mode=pad_mode)

    # Muestreo inverso (reverse diffusion). Solo se soporta OUVESDE, que es
    # la SDE empleada por el checkpoint WSJ0-REVERB; si en el futuro se usa
    # otro checkpoint con una SDE distinta (p. ej. SBVESDE), habría que
    # añadir esa rama siguiendo enhancement.py como referencia.
    if modelo.sde.__class__.__name__ != "OUVESDE":
        raise RuntimeError(
            f"SDE '{modelo.sde.__class__.__name__}' no soportado por este wrapper "
            "(solo probado con OUVESDE, la usada por el checkpoint WSJ0-REVERB)."
        )

    sampler = modelo.get_pc_sampler(
        "reverse_diffusion", SGMSE_CORRECTOR, Y.to(device),
        N=SGMSE_N, corrector_steps=SGMSE_CORRECTOR_STEPS, snr=SGMSE_SNR,
    )
    sample, _ = sampler()

    # Vuelta al dominio temporal y desnormalización
    audio_mejorado = modelo.to_audio(sample.squeeze(), T_orig) * norm_factor
    audio_mejorado_arr = audio_mejorado.detach().cpu().numpy().squeeze()

    pico_antes = np.max(np.abs(audio_mejorado_arr))
    audio_mejorado_arr = normalizar_pico(audio_mejorado_arr, pico_objetivo=0.95)

    guardar_audio(ruta_salida, audio_mejorado_arr, sr_modelo)

    return audio_mejorado_arr, sr_modelo, pico_antes
