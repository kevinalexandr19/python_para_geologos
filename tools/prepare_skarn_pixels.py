"""Recreate the browser-sized XGBoost lesson assets from the original image pairs."""
from pathlib import Path
from io import BytesIO

import numpy as np
import pandas as pd
from PIL import Image


def skarn_assets():
    root = Path(__file__).resolve().parents[1] / 'notebooks/files/xgboost'
    etiquetas = pd.read_excel(root / 'training_data.xlsx', dtype={'id': str})
    rng = np.random.default_rng(42)
    tablas = []
    for segmento, etiqueta in etiquetas[['id', 'class']].itertuples(index=False, name=None):
        with Image.open(root / 'training' / f'skarn_1_np_{segmento}.png') as np_img:
            np_segmento = np.array(np_img.convert('RGBA'))
        with Image.open(root / 'training' / f'skarn_1_nc_{segmento}.png') as nc_img:
            nc_segmento = np.array(nc_img.convert('RGBA'))
        assert np_segmento.shape == nc_segmento.shape
        assert np.array_equal(np_segmento[:, :, 3], nc_segmento[:, :, 3])
        opacos = np.flatnonzero(np_segmento[:, :, 3] == 255)
        indices = rng.choice(opacos, size=min(384, len(opacos)), replace=False)
        filas, cols = np.unravel_index(indices, np_segmento.shape[:2])
        colores = np.column_stack([np_segmento[filas, cols, :3], nc_segmento[filas, cols, :3]])
        tabla = pd.DataFrame(colores, columns=['R1', 'G1', 'B1', 'R2', 'G2', 'B2'])
        tabla = tabla.assign(imagen='skarn_1', segmento=segmento, fila=filas, columna=cols, etiqueta=etiqueta)
        tablas.append(tabla)
    tabla = pd.concat(tablas, ignore_index=True)
    vistas = {}
    for modo in ('np', 'nc'):
        with Image.open(root/'test'/f'skarn_2_{modo}.jpg') as im:
            vistas[modo] = np.array(im.convert('RGB'))[::12, ::12].copy()
    assert vistas['np'].shape == vistas['nc'].shape
    buffer = BytesIO()
    np.savez_compressed(buffer, **vistas)
    return {
        'files/xgboost/skarn_pixeles.csv': tabla.to_csv(index=False, lineterminator='\n').encode('utf8'),
        'files/xgboost/skarn_2_vistas.npz': buffer.getvalue(),
    }


if __name__ == '__main__':
    destino = Path(__file__).resolve().parents[1] / 'notebooks'
    for ruta, datos in skarn_assets().items():
        (destino / ruta).write_bytes(datos)
