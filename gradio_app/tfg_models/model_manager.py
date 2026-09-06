"""
Gestor central de modelos para la app.

Implementa el patrón acordado: un único modelo cargado en GPU/memoria a la vez.
Cuando se pide un modelo distinto al que ya está cargado, se libera el actual
antes de cargar el nuevo.

Nota de diseño respecto a `audio_utils_memoria_GPU.liberar_memoria_gpu`:
esa función usaba `inspect` para borrar la variable del propio notebook porque
en Colab el modelo vivía como variable global de cada celda. Aquí el modelo
vive como variable global de este módulo (`_modelo_actual`), así que no hace
falta ese truco de introspección de frame: basta con reasignar la variable
del módulo a None. El efecto (mover a CPU, borrar referencia, vaciar caché
CUDA) es el mismo.
"""

import gc
import torch

_modelo_actual = None
_categoria_actual = None
_variante_actual = None


def obtener_modelo(categoria: str, variante: str, funcion_carga):
    """
    Devuelve el modelo solicitado, cargándolo si hace falta.

    Args:
        categoria: nombre de la categoría de degradación (ej. 'denoising')
        variante: nombre de la variante/modelo (ej. 'deepfilternet')
        funcion_carga: función sin argumentos que carga y devuelve el modelo
                       (o un dict con el modelo y su estado, según el caso)

    Si la categoria+variante coincide con lo ya cargado, se reutiliza tal cual
    (no se recarga en cada petición). Si no coincide, se libera lo anterior
    primero.
    """
    global _modelo_actual, _categoria_actual, _variante_actual

    if (
        _modelo_actual is not None
        and _categoria_actual == categoria
        and _variante_actual == variante
    ):
        return _modelo_actual

    liberar_modelo_actual()

    _modelo_actual = funcion_carga()
    _categoria_actual = categoria
    _variante_actual = variante
    return _modelo_actual


def liberar_modelo_actual():
    """Libera el modelo actualmente cargado (si hay alguno) y vacía la caché CUDA."""
    global _modelo_actual, _categoria_actual, _variante_actual

    if _modelo_actual is not None:
        objeto = _modelo_actual
        # Si es un dict (como en denoising.py, que guarda modelo + df_state),
        # intentamos mover a CPU cualquier valor que tenga el método .to()
        candidatos = objeto.values() if isinstance(objeto, dict) else [objeto]
        for candidato in candidatos:
            try:
                if hasattr(candidato, "to"):
                    candidato.to("cpu")
            except Exception:
                pass
        del objeto

    _modelo_actual = None
    _categoria_actual = None
    _variante_actual = None

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
