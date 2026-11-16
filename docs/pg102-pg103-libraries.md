# Librerías de PG102 y PG103 en JupyterLite

Los notebooks usan el kernel Pyodide 0.27.6 existente. Las celdas de preparación instalan las dependencias del navegador; pip en Python local es un entorno independiente.

- Seaborn 0.13.2, Plotly 6.5.2, Geopy 2.4.1 y Openpyxl 3.1.5: versiones usadas en los ejemplos.
- Plotly necesita nbformat en el kernel para publicar figuras con el renderer MIME. Geopy necesita cargar explícitamente ssl de la distribución Pyodide, aunque aquí solo usemos distancias geodésicas.
- Pyrolite 0.3.6: compatible con Matplotlib 3.8.4 de este kernel. Las versiones posteriores exigen un Matplotlib más reciente.
- Para los ejemplos de Pyrolite se preparan explícitamente las dependencias de cálculo y gráficos. Se instalan Pyrolite y Numpydoc con `deps=False`: solo se usa `numpydoc.docscrape`, sin cargar la extensión Sphinx ni su cadena de herramientas de documentación. Esta preparación está validada para los ejemplos del notebook, no para construir la documentación de esas librerías.
- En PG103c los paquetes se preparan de uno en uno, mostrando el requisito actual. Esta secuencia se verificó en una sesión nueva de JupyterLite y evita el lote de descargas que falló durante la revisión.
- CoreLab 1.0.3: última publicación en PyPI verificada el 16 de noviembre de 2026. Se instala desde PyPI sin modificar sus archivos. En Pyodide, `corelab_compat.py` desactiva únicamente el decorador JIT de Numba durante el import, restaura `sys.modules` y proporciona el import de `display` ausente en el paquete. Los cálculos se ejecutan en Python. No equivale a soporte general de Numba en el navegador.
- Mplstereonet 0.6.3: PyPI ofrece solo un sdist. El archivo `notebooks/files/mplstereonet-0.6.3-py3-none-any.whl` se construyó sin cambios desde ese sdist oficial. Incluye su licencia y no requiere compilación nativa.

Fuente de Mplstereonet: https://files.pythonhosted.org/packages/ae/41/52c87d0c1b0799f5948f31bf63bc56350db9da9d1e8b6f56fa7ebe315069/mplstereonet-0.6.3.tar.gz

SHA-256 del sdist: `2ce630b14973a0b2f7d65cabb8d03fada2a8fda8b9b922d4cf335ecfcb442372`

Construcción con pip y setuptools/wheel instalados:

```sh
python -m pip wheel --no-cache-dir --no-deps --no-build-isolation mplstereonet-0.6.3.tar.gz --wheel-dir wheels
```

SHA-256 de la rueda distribuida: `3a1edd23e137d1c57985bdaca0146e5230f4f06117fb46c119a857b16b7b7a7d`

En JupyterLite se instala con micropip desde el sistema de archivos (`emfs:`), después de cargar Matplotlib y NumPy. Los mosaicos de los mapas base requieren conexión a Internet; los visores 3D de Plotly requieren WebGL habilitado. El HTML exportado con `include_plotlyjs=True` incluye el motor de Plotly.
