"""GAMMA: semivariogramas experimentales y modelos didácticos.

Copyright 2023, kevinalexandr19. Distribuido bajo la licencia MIT.
"""

import numpy as np


def _observations(values, coords):
    values = np.asarray(values, dtype=float)
    coords = np.asarray(coords, dtype=float)
    if values.ndim != 1 or values.size < 2:
        raise ValueError("Se requieren al menos dos valores en un vector.")
    if coords.ndim == 1:
        coords = coords[:, None]
    if coords.ndim != 2:
        raise ValueError("Usa coordenadas (n,), (n, d) o (d, n).")
    if coords.shape[0] != values.size and coords.shape[1] == values.size:
        coords = coords.T
    if coords.shape[0] != values.size or not 1 <= coords.shape[1] <= 3:
        raise ValueError("Cada valor debe tener una fila de 1 a 3 coordenadas.")
    if not np.isfinite(values).all() or not np.isfinite(coords).all():
        raise ValueError("Filtra las filas con valores o coordenadas no finitos.")
    return values, coords


def _pair_data(values, coords):
    # Conservamos el orden de los pares: primero (i, i+1), luego (i, i+2), etc.
    n = len(values)
    count = n * (n - 1) // 2
    i = np.empty(count, dtype=np.intp)
    j = np.empty_like(i)
    start = 0
    for step in range(1, n):
        stop = start + n - step
        i[start:stop] = np.arange(n - step)
        j[start:stop] = np.arange(step, n)
        start = stop
    delta = coords[j] - coords[i]
    return i, j, np.linalg.norm(delta, axis=1), (values[j] - values[i]) ** 2, delta


def gam1d(values, coords):
    """Devuelve las distancias y diferencias al cuadrado de los pares 1D."""
    values, coords = _observations(values, coords)
    if coords.shape[1] != 1:
        raise ValueError("gam1d requiere una coordenada por muestra.")
    _, _, lags, sqvalues, _ = _pair_data(values, coords)
    return lags, sqvalues


def gam2d(values, coords):
    """Devuelve distancias, diferencias al cuadrado y azimut desde el norte."""
    values, coords = _observations(values, coords)
    if coords.shape[1] != 2:
        raise ValueError("gam2d requiere dos coordenadas por muestra.")
    _, _, lags, sqvalues, delta = _pair_data(values, coords)
    angles = np.degrees(np.arctan2(delta[:, 0], delta[:, 1]))
    return lags, sqvalues, angles


def linear(h, c0, c, a):
    """Modelo sin meseta: c/a es la pendiente para h > 0."""
    h = np.asarray(h)
    return np.where(h == 0, 0.0, c0 + c * h / a)


def exponential(h, c0, c, a):
    """Modelo exponencial, con a como escala interna."""
    h = np.asarray(h)
    return np.where(h == 0, 0.0, c0 - c * np.expm1(-h / a))


def spherical(h, c0, c, a):
    """Modelo esférico, con meseta a partir de la distancia a."""
    h = np.asarray(h)
    r = np.minimum(h / a, 1.0)
    return np.where(h == 0, 0.0, c0 + c * (1.5 * r - 0.5 * r**3))


def gaussian(h, c0, c, a):
    """Modelo gaussiano, con a como escala interna."""
    h = np.asarray(h)
    return np.where(h == 0, 0.0, c0 - c * np.expm1(-(h / a)**2))


MODELS = {"linear": linear, "exponential": exponential, "spherical": spherical, "gaussian": gaussian}
RANGE_FACTORS = {"linear": 1.0, "exponential": np.log(20), "spherical": 1.0,
                 "gaussian": np.sqrt(np.log(20))}


def _show_in_widget(fig, image, status, message):
    """Update widget traits directly, including in browser kernel callbacks."""
    from html import escape
    from io import BytesIO
    import matplotlib.pyplot as plt

    with BytesIO() as buffer:
        fig.savefig(buffer, format="png", dpi=105, bbox_inches="tight", facecolor="white")
        image.value = buffer.getvalue()
    plt.close(fig)
    status.value = f"<pre>{escape(message)}</pre>"


def r2_score(gamma, prediction):
    observed = np.asarray(gamma)
    total = np.sum((observed - observed.mean()) ** 2)
    return None if total == 0 else float(1 - np.sum((observed - prediction) ** 2) / total)


def mse_score(gamma, prediction):
    return float(np.mean((np.asarray(gamma) - prediction) ** 2))


class Variogram:
    """Agrupa pares de coordenadas 1D, 2D o 3D en clases de distancia.

    lagDist fija el paso entre centros. La primera clase es (0, 1.5*lagDist];
    luego cada clase tiene ancho lagDist y su límite superior es inclusivo.
    max_lag, si se indica, limita la distancia real del par, no la nominal.
    Los pares coincidentes quedan fuera de las clases positivas. Para 2D,
    azimuth se mide desde +Y hacia +X y representa un eje (módulo 180°).
    El número de pares crece como n*(n-1)/2. Usar subconjuntos para datos grandes.
    """

    def __init__(self, values, coords, lagDist=None, *, max_lag=None,
                 azimuth=None, angle_tolerance=22.5):
        self.values, points = _observations(values, coords)
        self.n, d = points.shape
        self.coords = points[:, 0] if d == 1 else points.T
        self.dim = f"{d}D"
        self.var = float(self.values.var())
        self.max_lag = None if max_lag is None else self._positive(max_lag, "max_lag")
        i, j, lags, squared, delta = _pair_data(self.values, points)
        order = np.argsort(lags, kind="stable")
        self.lags, self.sqvalues = lags[order], squared[order]
        self._i, self._j = i[order], j[order]
        self.coincident_pairs = int(np.count_nonzero(self.lags == 0))
        self.angles = None
        if d == 2:
            self.angles = np.degrees(np.arctan2(delta[order, 0], delta[order, 1]))

        usable = self.lags > 0
        if self.max_lag is not None:
            usable &= self.lags <= self.max_lag
        self.azimuth = None
        self.angle_tolerance = None
        if azimuth is not None:
            if d != 2:
                raise ValueError("El filtro por azimut requiere coordenadas 2D.")
            azimuth, tolerance = float(azimuth), float(angle_tolerance)
            if not np.isfinite(azimuth) or not 0 <= tolerance <= 90:
                raise ValueError("Usa un azimut finito y una tolerancia entre 0 y 90°.")
            self.azimuth, self.angle_tolerance = azimuth % 180, tolerance
            angular_gap = (self.angles - self.azimuth + 90) % 180 - 90
            usable &= np.abs(angular_gap) <= tolerance
        self._usable = usable
        self.lagDistSelector = LagDistSelector(self)

        if lagDist is None:
            # La mediana de los vecinos más próximos sirve como punto de partida.
            # Un par en la misma coordenada aporta 0 m, y se excluye de esta regla.
            nearest = np.full(self.n, np.inf)
            positive = lags > 0
            np.minimum.at(nearest, i[positive], lags[positive])
            np.minimum.at(nearest, j[positive], lags[positive])
            finite = nearest[np.isfinite(nearest)]
            if not len(finite):
                raise ValueError("Se requieren al menos dos coordenadas distintas.")
            lagDist = np.median(finite)
        self.updateLagDist(lagDist)

    @staticmethod
    def _positive(value, name):
        value = float(value)
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"{name} debe ser finito y mayor que 0.")
        return value

    def updateLagDist(self, lagDist=None):
        """Recalcula las clases usando los mismos pares e invalida el ajuste previo."""
        step = self._positive(self.lagDistSelector.lagDist if lagDist is None else lagDist,
                              "lagDist")
        distances, square = self.lags[self._usable], self.sqvalues[self._usable]
        if len(distances) and distances[-1] / 2**52 >= step:
            raise ValueError("lagDist es demasiado pequeño para estas distancias.")
        scaled = distances / step
        classes = np.maximum(1, np.ceil(scaled - 0.5)).astype(np.int64)
        groups, inverse, count = np.unique(classes, return_inverse=True, return_counts=True)
        self.lagDist = self.lagDistSelector.lagDist = step
        self.h = groups * step
        self.lower = np.where(groups == 1, 0.0, (groups - 0.5) * step)
        self.bins = (groups + 0.5) * step
        self.mean_lag = np.bincount(inverse, weights=distances) / count
        self.gamma = 0.5 * np.bincount(inverse, weights=square) / count
        self.pairs = count
        self._class_centers = self.h[inverse]
        self.model = self.current_model = None
        return self

    def pair_table(self):
        """Muestra las filas, distancias y contribuciones de los pares incluidos."""
        import pandas as pd

        used = self._usable
        return pd.DataFrame({"i": self._i[used], "j": self._j[used], "h_real": self.lags[used],
                             "diferencia²": self.sqvalues[used], "h_clase": self._class_centers,
                             "semivarianza_par": self.sqvalues[used] / 2})

    def lag_table(self):
        """Devuelve centros nominales, distancias medias, conteos y semivarianzas."""
        import pandas as pd

        return pd.DataFrame({"h": self.h, "desde": self.lower, "hasta": self.bins,
                             "h_media": self.mean_lag, "pares": self.pairs, "gamma": self.gamma})

    def __str__(self):
        summary = (f"Variograma experimental {self.dim}\n"
                   f"Muestras: {self.n} | Pares usados: {self.pairs.sum():,} de {len(self.lags):,}\n"
                   f"Clases con datos: {len(self.h)} | Paso: {self.lagDist:g}\n")
        if self.model is not None:
            m = self.model
            reach = "sin meseta" if m["a"] is None else f"{m['a']:.4g}"
            summary += f"Modelo: {m['model']} | C0: {m['c0']:.4g} | C: {m['c']:.4g} | Alcance: {reach}"
        return summary.strip()

    def _fit_indices(self, n_points, min_pairs):
        if int(min_pairs) != min_pairs or min_pairs < 1:
            raise ValueError("min_pairs debe ser un entero positivo.")
        eligible = np.flatnonzero(self.pairs >= min_pairs)
        if n_points is None:
            return eligible
        if int(n_points) != n_points or not 1 <= n_points <= len(eligible):
            raise ValueError("n_points debe caber en las clases con suficientes pares.")
        return eligible[:int(n_points)]

    def _save_model(self, name, popt, index, min_pairs):
        c0, c, scale = map(float, popt)
        prediction = MODELS[name](self.mean_lag[index], c0, c, scale)
        has_sill = name != "linear"
        self.model = self.current_model = {
            "model": name, "n_points": len(index), "min_pairs": int(min_pairs),
            "c0": c0, "c": c, "a": float(scale * RANGE_FACTORS[name]) if has_sill else None,
            "sill": c0 + c if has_sill else None, "popt": [c0, c, scale],
            "r2": r2_score(self.gamma[index], prediction),
            "mse": mse_score(self.gamma[index], prediction),
        }
        return self.model

    def fit(self, model="spherical", n_points=None, *, min_pairs=1):
        """Ajusta un modelo a la distancia media de cada clase, ponderada por pares."""
        from scipy.optimize import curve_fit, nnls

        if model not in MODELS:
            raise ValueError(f"Modelo desconocido. Usa uno de {list(MODELS)}.")
        index = self._fit_indices(n_points, min_pairs)
        if len(index) < (2 if model == "linear" else 3):
            raise ValueError("Hay muy pocas clases para ajustar este modelo.")
        h, observed = self.mean_lag[index], self.gamma[index]
        weights = np.sqrt(self.pairs[index])
        if model == "linear":
            # Usamos una escala fija para evitar ajustar por separado dos
            # parámetros que en la recta solo aparecen como c/a.
            scale = float(h[-1])
            design = np.column_stack([np.ones(len(h)), h / scale])
            c0, c = nnls(design * weights[:, None], observed * weights)[0]
            popt = [c0, c, scale]
        elif np.all(observed == 0):
            popt = [0, 0, float(h[-1] / 2)]
        else:
            amplitude = observed.max()
            # Normalizamos h y gamma para evitar mal condicionamiento
            # cuando las coordenadas y las semivarianzas tienen escalas distintas.
            distance = h[-1]
            start = [min(observed[0] / amplitude, 0.25), 0.8, 0.5]
            scaled, _ = curve_fit(MODELS[model], h / distance, observed / amplitude,
                                  sigma=1 / weights, p0=start,
                                  bounds=([0, 0, 1e-6], [np.inf, np.inf, np.inf]),
                                  maxfev=6000)
            popt = scaled * [amplitude, amplitude, distance]
        return self._save_model(model, popt, index, min_pairs)

    def set_model(self, model, c0, c, a, n_points=None, *, min_pairs=1):
        """Fija parámetros. a es el alcance al 95% en exponencial y gaussiano."""
        if model not in MODELS:
            raise ValueError(f"Modelo desconocido. Usa uno de {list(MODELS)}.")
        c0, c = float(c0), float(c)
        if not np.isfinite([c0, c]).all() or min(c0, c) < 0:
            raise ValueError("C0 y C deben ser finitos y no negativos.")
        a = self._positive(a, "a")
        index = self._fit_indices(n_points, min_pairs)
        if len(index) == 0:
            raise ValueError("No hay clases con suficientes pares.")
        return self._save_model(model, [c0, c, a / RANGE_FACTORS[model]], index, min_pairs)

    def predict(self, h):
        """Evalúa el modelo guardado en distancias no negativas."""
        if self.model is None:
            raise ValueError("Ajusta o define un modelo primero.")
        h = np.asarray(h, dtype=float)
        if not np.isfinite(h).all() or np.any(h < 0):
            raise ValueError("Las distancias deben ser finitas y no negativas.")
        return MODELS[self.model["model"]](h, *self.model["popt"])

    def plot(self, *, distance_unit="m", value_unit="unidad"):
        """Dibuja semivarianzas y conteos de pares con el mismo eje de distancia."""
        import matplotlib.pyplot as plt

        if not len(self.h):
            raise ValueError("No quedan pares con estos filtros.")
        fig, (ax, counts) = plt.subplots(2, 1, figsize=(8.8, 5.6), sharex=True,
                                        gridspec_kw={"height_ratios": [3, 1]})
        ax.scatter(self.mean_lag, self.gamma, facecolors="white", edgecolors="#0072B2",
                   s=39, label="Clases experimentales", zorder=3)
        if self.model is not None:
            m = self.model
            used = self._fit_indices(m["n_points"], m["min_pairs"])
            ax.scatter(self.mean_lag[used], self.gamma[used], color="#0072B2", s=19,
                       label="Clases usadas en el ajuste", zorder=4)
            last = self.mean_lag[used[-1]]
            end = self.mean_lag[-1]
            h = np.linspace(last * 1e-6, last, 200)
            ax.plot(h, self.predict(h), color="#B84D12", lw=2, label=m["model"])
            if end > last:
                extrapolation = np.linspace(last, end, 160)
                ax.plot(extrapolation, self.predict(extrapolation), "--", color="#B84D12",
                        label="Fuera del ajuste")
            ax.scatter(0, m["c0"], facecolor="white", edgecolor="#B84D12", zorder=5)
            ax.scatter(0, 0, color="#B84D12", s=17, zorder=5)
        ax.set(title=f"Variograma experimental {self.dim}", ylabel=f"Semivarianza ({value_unit})²")
        ax.set_ylim(bottom=0)
        ax.legend(frameon=False, fontsize=9)
        counts.bar(self.mean_lag, self.pairs, width=0.62 * self.lagDist, color="#608DA4")
        counts.set(xlabel=f"Distancia media del par ({distance_unit})", ylabel="Pares")
        counts.set_xlim(left=-0.13 * self.lagDist)
        for axis in (ax, counts):
            axis.grid(alpha=.20)
            axis.set_axisbelow(True)
        fig.tight_layout()
        return fig, (ax, counts)

    def selectLagDist(self):
        """Abre el control interactivo para cambiar el paso de las clases."""
        return self.lagDistSelector.lagDistWidget()

    def fitting(self, *, distance_unit="m", value_unit="unidad"):
        """Abre controles de ajuste automático y manual sin calcular todos los modelos."""
        import ipywidgets as widgets
        from IPython.display import display
        from html import escape

        if len(self.h) < 2 and self.model is None:
            raise ValueError("Se requieren al menos dos clases para usar el ajuste.")
        saved = self.model or self.fit("spherical" if len(self.h) >= 3 else "linear")
        min_pairs = saved["min_pairs"]
        eligible = int((self.pairs >= min_pairs).sum())
        name = widgets.Dropdown(options=list(MODELS), value=saved["model"], description="Modelo")
        n = widgets.IntSlider(min=1, max=eligible, value=saved["n_points"],
                              continuous_update=False, description="Clases")
        auto = widgets.Checkbox(value=True, description="Autoajuste")
        c0 = widgets.FloatText(value=saved["c0"], description="Pepita C0", disabled=True)
        c = widgets.FloatText(value=saved["c"], description="Parcial C", disabled=True)
        a = widgets.FloatText(value=saved["a"] or saved["popt"][2], description="a", disabled=True)
        run = widgets.Button(description="Aplicar modelo", icon="check", button_style="success")
        image = widgets.Image(format="png", layout=widgets.Layout(width="100%", max_width="900px"))
        status = widgets.HTML()

        def render():
            fig, _ = self.plot(distance_unit=distance_unit, value_unit=value_unit)
            _show_in_widget(fig, image, status, str(self))

        def apply(_):
            try:
                if auto.value:
                    m = self.fit(name.value, n.value, min_pairs=min_pairs)
                    c0.value, c.value = m["c0"], m["c"]
                    a.value = m["a"] or m["popt"][2]
                else:
                    self.set_model(name.value, c0.value, c.value, a.value, n.value,
                                   min_pairs=min_pairs)
                render()
            except (ValueError, RuntimeError) as error:
                status.value = f"<pre>{escape(str(error))}</pre>"

        run.on_click(apply)

        def toggle_manual(change):
            for field in (c0, c, a):
                field.disabled = change["new"]

        auto.observe(toggle_manual, names="value")
        render()
        panel = widgets.VBox([widgets.HBox([name, auto]), n, c0, c, a, run, status, image])
        display(panel)
        return panel


class LagDistSelector:
    """Interfaz ligera para recalcular clases sin volver a generar pares."""

    def __init__(self, variogram):
        self.Variogram = variogram
        self.lagDist = None

    @property
    def lags(self):
        return self.Variogram.lags

    def lagDistPlot(self, lagDist, distance):
        import matplotlib.pyplot as plt

        vag = self.Variogram
        step = vag._positive(lagDist, "lagDist")
        cutoff = vag._positive(distance, "distance")
        used = vag._usable
        near = vag.lags[used & (vag.lags <= cutoff)]
        fig, ax = plt.subplots(figsize=(8.8, 2.8))
        ax.hist(near, bins=120, range=(0, cutoff), color="#0072B2")
        for k in range(min(50, max(0, int(cutoff / step - 0.5)))):
            ax.axvline((k + 1.5) * step, color="#B84D12", alpha=.7, lw=.9)
        ax.set(xlim=(0, cutoff), xlabel="Distancia real entre pares",
               ylabel="Pares", title=f"Distancias y límites de clase · paso {step:g}")
        fig.tight_layout()
        return fig, ax

    def lagDistWidget(self):
        import ipywidgets as widgets
        from IPython.display import display
        from html import escape

        v = self.Variogram
        step = widgets.FloatText(value=v.lagDist, description="Paso")
        apply_step = widgets.Button(description="Aplicar paso", icon="check", button_style="success")
        image = widgets.Image(format="png", layout=widgets.Layout(width="100%", max_width="900px"))
        status = widgets.HTML()

        def show():
            fig, _ = self.lagDistPlot(v.lagDist, min(v.lags.max(), 8 * v.lagDist))
            _show_in_widget(fig, image, status, str(v))

        def apply(_):
            try:
                v.updateLagDist(step.value)
                show()
            except ValueError as error:
                status.value = f"<pre>{escape(str(error))}</pre>"

        apply_step.on_click(apply)
        show()
        panel = widgets.VBox([step, apply_step, status, image])
        display(panel)
        return panel
