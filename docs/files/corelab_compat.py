"""Adaptador del curso para CoreLab 1.0.3 publicado en PyPI.

En Pyodide, conserva el cálculo Python de CoreLab sin aceleración Numba.
El módulo numba temporal solo se usa durante el import y se retira después.
No cambia la trayectoria ni las fórmulas del paquete instalado.
"""
import importlib
import sys
import types
from IPython.display import display


def cargar_corelab():
    if sys.platform == "emscripten" and "corelab.core3d" not in sys.modules:
        acelerador = types.ModuleType("numba")
        acelerador.jit = lambda **opciones: lambda funcion: funcion
        anterior = sys.modules.get("numba")
        sys.modules["numba"] = acelerador
        try:
            modulo = importlib.import_module("corelab.core3d")
        finally:
            if anterior is None:
                sys.modules.pop("numba", None)
            else:
                sys.modules["numba"] = anterior
    else:
        modulo = importlib.import_module("corelab.core3d")
    # La versión publicada usa display sin importarlo en core3d.
    modulo.display = display
    if sys.platform == "emscripten":
        modulo.tqdm.monitor_interval = 0
    return modulo.DrillData
