# Figuras didácticas originales

Las figuras `bootstrap.png`, `etapas_analisis.png`, `boxplot.png` y
`box_violinplot.png` se generan con `python tools/generate_teaching_figures.py`.
Requieren NumPy y Matplotlib; no añaden dependencias a los notebooks.
El bootstrap reproduce la semilla y la secuencia de extracciones del notebook.
Los dos gráficos de distribución usan las mismas 20 observaciones didácticas.
Los cuartiles usan la interpolación lineal predeterminada de NumPy/Matplotlib.

`montecarlo_simulation.png` es una ilustración original generada con la herramienta
ImageGen integrada. No representa resultados de una simulación real.
El notebook presenta el PNG sin modificarlo en un marco HTML adaptable;
su proporción 2998:949 produce exactamente la mitad de la altura de la imagen
anterior (1499:949), al mismo ancho de visualización.

Después de regenerar, actualizar los adjuntos de las celdas y sincronizar
`content/`, `docs/files/` y los catálogos de JupyterLite.

## Prompt de la ilustración

Use case: illustration-story / scientific-educational. Create an ORIGINAL artistic horizontal banner for a Spanish geology programming notebook, replacing a stock Monte Carlo illustration. This is a NEW composition, do not copy any existing image. Target very wide low-height banner, approximately 3.16:1 aspect ratio (ideally 2998 x 949 pixels). It will display at the original width but HALF the original height. All essential content must fit a low, wide banner, with generous safe margins. A refined editorial illustration of two ivory dice, a few teal and warm-gold casino chips and two tasteful playing cards on a deep midnight-navy background. Arrange the objects as a coherent restrained still life with soft 3D shading, crisp silhouettes and a premium scientific-book aesthetic. A subtle trail of repeated small dice/chips may suggest repeated random trials, without a numerical plot, equations or promises about real outcomes. Composition: compact illustration group on the left third, clear highly legible Spanish typography on the remaining area, balanced open space, no clutter. Main title, exact verbatim: "Simulación de Montecarlo". Short description below it, exact verbatim: "Repetimos escenarios con variables aleatorias para explorar posibles resultados y su incertidumbre." Small final sentence, exact verbatim: "Los resultados dependen de los supuestos del modelo." Large title in off-white, readable supporting text, restrained teal and gold accents. Typography must be correctly accented, clean and easy to read when the banner is shown in a notebook at about 900 pixels wide. Do not write any other text. No logos, no trademarks, no watermark, no Investopedia, no Python para Geólogos banner, no decorative axes or fake statistics. This is conceptual artwork, not a data chart.
