import gc
import inspect
import torch


def liberar_memoria_gpu(modelo=None, nombre_variable=None):
    """
    Libera la memoria GPU ocupada por un modelo de PyTorch.

    Args:
        modelo: objeto del modelo (opcional), usado solo para moverlo a CPU antes de liberar.
        nombre_variable (str, opcional): nombre EXACTO de la variable en el notebook que
            contiene el modelo (ej. 'modelo_dfn'). Sin esto, borrar la referencia local
            'modelo' dentro de esta función no libera nada de verdad: el notebook conserva
            su propia referencia al objeto, así que el recuento de referencias nunca llega
            a 0. Con nombre_variable, se localiza el frame de quien llamó a la función
            (el notebook) mediante `inspect` y se borra también esa variable ahí.

    Ejemplo de uso: liberar_memoria_gpu(modelo_dfn, 'modelo_dfn')
    """
    if modelo is not None:
        try:
            modelo.to('cpu')
        except Exception:
            pass
        del modelo

    if nombre_variable is not None:
        frame_llamador = inspect.currentframe().f_back
        if nombre_variable in frame_llamador.f_globals:
            del frame_llamador.f_globals[nombre_variable]
        else:
            print(f"Aviso: no se encontró una variable llamada '{nombre_variable}' "
                  "en el notebook, no se ha borrado nada por ese lado.")
    else:
        print('Aviso: sin nombre_variable, solo se libera la referencia local a esta '
              'función. Si el modelo sigue asignado a una variable en el notebook '
              '(ej. modelo_dfn), la GPU no se liberará del todo. Llama a esta función '
              "como liberar_memoria_gpu(modelo_dfn, 'modelo_dfn') para liberarla de verdad.")

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(f'Memoria GPU liberada. Uso actual: {torch.cuda.memory_allocated() / 1e9:.2f} GB')
    else:
        print('No hay GPU disponible en esta sesión (Entorno de ejecución > Cambiar tipo de entorno > GPU).')
