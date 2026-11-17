"""Reproducible original teaching figures; never edits the generated artwork."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.cbook import boxplot_stats

OUT = Path(__file__).resolve().parents[1] / 'notebooks' / 'resources'
OUT.mkdir(parents=True, exist_ok=True)
NAVY, TEAL, GOLD, GREY = '#19334a', '#087e8b', '#b97913', '#526575'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 12,
                     'text.color': NAVY, 'axes.labelcolor': NAVY,
                     'xtick.color': GREY, 'ytick.color': GREY,
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'savefig.facecolor': 'white', 'axes.titleweight': 'bold'})

def save(fig, name):
    fig.savefig(OUT / name, dpi=150)
    plt.close(fig)

# Reproduce the notebook RNG sequence exactly: ten original observations,
# then 1000 samples of ten draws with replacement.
rng = np.random.default_rng(42)
data = rng.normal(size=10)
ids = rng.choice(np.arange(10), size=(1000, 10), replace=True)
means = data[ids].mean(axis=1)
check_rng = np.random.default_rng(42)
check_data = np.array([check_rng.normal() for _ in range(10)])
check_bootstrap = np.array([[check_rng.choice(check_data) for _ in range(10)] for _ in range(1000)])
assert np.array_equal(data[ids], check_bootstrap)
fig = plt.figure(figsize=(12, 5.6))
fig.suptitle('Bootstrap: reutilizamos observaciones para explorar la incertidumbre',
             x=.05, ha='left', y=.965, fontsize=16, weight='bold')
fig.text(.05, .90, 'Cada círculo identifica una observación; su número se conserva cuando vuelve a aparecer.',
         fontsize=11, color=GREY)
axes = [fig.add_axes([.05, .23, .22, .56]), fig.add_axes([.33, .23, .34, .56]),
        fig.add_axes([.75, .27, .22, .46])]
cmap = plt.get_cmap('viridis')
colors = cmap(np.linspace(.13, .85, 10))
axes[0].set(xlim=(-.65, 4.65), ylim=(-.8, 2.2))
axes[0].set_title('1. Muestra original', loc='left', fontsize=13, pad=18)
for i in range(10):
    x, y = i % 5, 1.15 - (i // 5)*.82
    axes[0].scatter(x, y, s=440, color=colors[i], edgecolor='white', zorder=2)
    axes[0].text(x, y, str(i+1), color='white' if i<6 else NAVY,
                 ha='center', va='center', fontsize=11, weight='bold')
axes[0].text(2, -.42, 'n = 10 observaciones\nNo añadimos datos nuevos.',
             ha='center', va='center', fontsize=11)
axes[1].set(xlim=(-.7, 12.5), ylim=(-.9, 3.5))
axes[1].set_title('2. Remuestreos con reemplazo', loc='left', fontsize=13, pad=18)
for row in range(3):
    y = 2.6-row*1.02
    for j, k in enumerate(ids[row]):
        axes[1].scatter(j, y, s=190, color=colors[k], edgecolor='white', zorder=2)
        axes[1].text(j, y, str(k+1), color='white' if k<6 else NAVY,
                     ha='center', va='center', fontsize=9, weight='bold')
    axes[1].text(10.1, y, f'→ {means[row]:.2f}', fontsize=11, va='center', color=TEAL)
axes[1].text(4.5, -.62, '10 extracciones por fila; algunas se repiten.\nMostramos 3 de los 1000 remuestreos.',
             ha='center', va='center', fontsize=10.5)
for ax in axes[:2]: ax.axis('off')
axes[2].hist(means, bins=20, color=TEAL, edgecolor='white', linewidth=.7)
axes[2].axvline(data.mean(), color=GOLD, ls='--', lw=2)
axes[2].set_title('3. Una media por fila', loc='left', fontsize=13, pad=20)
axes[2].set(xlabel='Media del remuestreo', ylabel='N.º de remuestreos')
axes[2].tick_params(labelsize=10)
for left, right, y in [(.28, .32, .53), (.68, .73, .72)]:
    fig.add_artist(FancyArrowPatch((left,y),(right,y), transform=fig.transFigure,
                                  arrowstyle='-|>', mutation_scale=18, color=GREY))
fig.text(.05, .09, 'La dispersión de las medias aproxima la incertidumbre de la media, bajo los supuestos del bootstrap.',
         fontsize=11)
fig.text(.05, .045, 'Semilla 42, n = 10 y m = 1000. Línea dorada: media original. La distribución no tiene por qué ser normal.',
         fontsize=10, color=GREY)
save(fig, 'bootstrap.png')

# Original Spanish conceptual diagram; stages are questions, not a hierarchy
# claiming that more complicated analyses are necessarily more valuable.
fig, ax = plt.subplots(figsize=(12, 5.3))
fig.subplots_adjust(left=.025, right=.975, top=.98, bottom=.02)
ax.set(xlim=(0, 12), ylim=(0, 5.3)); ax.axis('off')
ax.text(.25, 4.86, 'Cuatro preguntas para orientar el análisis de datos', fontsize=20, weight='bold')
ax.text(.25, 4.36, 'Ejemplo geológico: comprender las leyes de un yacimiento y decidir dónde muestrear.', fontsize=12, color=GREY)
cards = [
    ('DESCRIPTIVO', '¿Qué ocurrió?', 'Resumir los datos\nobservados.', 'Mapas y estadísticas\nde leyes por zona.'),
    ('DIAGNÓSTICO', '¿Por qué ocurrió?', 'Explorar relaciones y\nposibles explicaciones.', 'Contrastar litología,\nalteración y leyes.'),
    ('PREDICTIVO', '¿Qué podría ocurrir?', 'Estimar resultados\ny su incertidumbre.', 'Predecir leyes en zonas\nsin muestras y validar.'),
    ('PRESCRIPTIVO', '¿Qué conviene hacer?', 'Comparar decisiones\ncon objetivos y restricciones.', 'Priorizar muestreo según\ncosto e incertidumbre.')]
for i, (name, question, action, example) in enumerate(cards):
    x = .25 + i*2.93
    ax.add_patch(FancyBboxPatch((x, 1.08), 2.68, 2.72, boxstyle='round,pad=.02,rounding_size=.08',
                                facecolor='#f2f7f8', edgecolor='#cbdce2', linewidth=1.2))
    ax.text(x+.16, 3.47, name, fontsize=11, color=TEAL, weight='bold')
    ax.text(x+.16, 2.98, question, fontsize=13, weight='bold')
    ax.text(x+.16, 2.36, action, fontsize=10.5, linespacing=1.6)
    ax.text(x+.16, 1.52, example, fontsize=10.5, color=GREY, linespacing=1.6)
ax.text(.25, .59, 'La pregunta define el método. Podemos volver a los datos y revisar el análisis en cualquier etapa.', fontsize=12)
ax.text(.25, .15, 'Las relaciones no prueban causalidad; las predicciones y decisiones dependen de los datos y de los supuestos.', fontsize=10.5, color=GREY)
save(fig, 'etapas_analisis.png')

# Same didactic observations for the boxplot and violin comparison.
values = np.array([2.1,2.3,2.4,2.5,2.6,2.7,2.8,2.9,3.1,3.2,
                   3.3,3.5,3.7,3.9,4.1,4.4,4.6,4.9,5.2,8.5])
stats = boxplot_stats(values, whis=1.5)[0]
q1, med, q3, iqr = (stats[k] for k in ['q1','med','q3','iqr'])
low, high = q1-1.5*iqr, q3+1.5*iqr
assert np.isclose(iqr, 1.5)
assert stats['whislo'] == 2.1 and stats['whishi'] == 5.2
assert list(stats['fliers']) == [8.5]
fig, ax = plt.subplots(figsize=(12, 5.5))
fig.subplots_adjust(left=.07, right=.975, bottom=.27, top=.83)
fig.suptitle('Boxplot: cuartiles, bigotes y posibles valores atípicos', fontsize=18, weight='bold', y=.965)
ax.boxplot(values, vert=False, positions=[0], widths=.52, patch_artist=True,
           boxprops={'facecolor':'#bce0e3','edgecolor':TEAL,'linewidth':2},
           medianprops={'color':NAVY,'linewidth':2.5},
           whiskerprops={'color':TEAL,'linewidth':2}, capprops={'color':TEAL,'linewidth':2},
           flierprops={'marker':'o','markerfacecolor':GOLD,'markeredgecolor':GOLD,'markersize':8})
for fence, label in [(low, 'Q1 − 1.5 × RIC'), (high, 'Q3 + 1.5 × RIC')]:
    ax.axvline(fence, ymin=.23, ymax=.83, color=GOLD, ls='--', lw=1.4)
    ax.text(fence, -1.0, f'{label}\n{fence:.3f}', ha='center', va='top', color=GOLD, fontsize=11)
for q, label, y in [(q1,'Q1',.93), (med,'Mediana',1.4), (q3,'Q3',.93)]:
    ax.annotate(f'{label}\n{q:.3f}', xy=(q,.26), xytext=(q,y), ha='center', fontsize=11,
                arrowprops={'arrowstyle':'-','color':GREY})
ax.annotate('Bigote inferior\n2.1', xy=(2.1,0), xytext=(1.3,.9), ha='center', fontsize=11,
            arrowprops={'arrowstyle':'->','color':GREY})
ax.annotate('Bigote superior\n5.2', xy=(5.2,0), xytext=(5.3,1.1), ha='center', fontsize=11,
            arrowprops={'arrowstyle':'->','color':GREY})
ax.annotate('Posible valor atípico\n8.5', xy=(8.5,0), xytext=(8.15,1.05), ha='center', fontsize=11,
            arrowprops={'arrowstyle':'->','color':GREY})
ax.scatter(values, np.full_like(values,-.67), s=24, color=TEAL)
ax.set(xlim=(0,9.2), ylim=(-1.5,1.7), yticks=[], xlabel='Concentración de Cu (%) — datos didácticos')
ax.spines[['left','bottom']].set_visible(False)
ax.text(3.55,-.38, 'RIC = Q3 − Q1 = 1.500', ha='center', fontsize=10.5, color=TEAL)
fig.text(.07,.11, 'Los bigotes llegan al dato más extremo dentro de los límites discontinuos; no necesariamente a esos límites.', fontsize=11)
fig.text(.07,.055, 'Los 20 puntos inferiores son las observaciones. Un valor atípico merece revisión; no implica un error.', fontsize=11, color=GREY)
save(fig, 'boxplot.png')

fig, axes = plt.subplots(1,2, figsize=(11.5, 5.5), sharey=True)
fig.subplots_adjust(left=.09, right=.97, bottom=.24, top=.79, wspace=.24)
fig.suptitle('Dos vistas de las mismas 20 observaciones', y=.965, fontsize=18, weight='bold')
fig.text(.5,.88,'El boxplot resume posiciones; el violín añade una estimación suavizada de la densidad.', ha='center', fontsize=12)
axes[0].boxplot(values, widths=.38, patch_artist=True,
                boxprops={'facecolor':'#bce0e3','edgecolor':TEAL},
                medianprops={'color':NAVY,'linewidth':2},
                whiskerprops={'color':TEAL},capprops={'color':TEAL},
                flierprops={'marker':'o','markerfacecolor':GOLD,'markeredgecolor':GOLD})
parts=axes[1].violinplot(values, positions=[1], widths=.7, showextrema=False, bw_method=.35)
for body in parts['bodies']: body.set(facecolor=TEAL, edgecolor=TEAL, alpha=.3)
axes[1].vlines(1,q1,q3,color=NAVY,lw=7,zorder=3)
axes[1].scatter([1],[med],facecolor='white',edgecolor=NAVY,s=50,zorder=4)
for ax, title in zip(axes,['Boxplot','Violín + cuartiles']):
    ax.set(title=title, xlim=(.4,1.65), ylim=(0,9.3), xticks=[])
    ax.scatter(np.full(len(values),.55),values,s=16,color=GREY,alpha=.75,zorder=3)
    ax.grid(axis='y',alpha=.18); ax.set_axisbelow(True)
    ax.spines[['bottom']].set_visible(False)
axes[0].set_ylabel('Concentración de Cu (%) — datos didácticos')
axes[1].annotate('Q3',xy=(1,q3),xytext=(1.31,q3+.35),fontsize=10,arrowprops={'arrowstyle':'-','color':GREY})
axes[1].annotate('Mediana',xy=(1,med),xytext=(1.27,med-.15),fontsize=10,arrowprops={'arrowstyle':'-','color':GREY})
axes[1].annotate('Q1',xy=(1,q1),xytext=(1.31,q1-.8),fontsize=10,arrowprops={'arrowstyle':'-','color':GREY})
fig.text(.09,.13,'Más ancho = mayor densidad estimada en ese rango. El ancho depende del suavizado elegido.',fontsize=11)
fig.text(.09,.065,'Los puntos representan los mismos datos en ambas figuras; la curva no añade observaciones.',fontsize=11,color=GREY)
save(fig, 'box_violinplot.png')

(OUT/'figure-statistics.json').write_text(json.dumps({
    'bootstrap': {'seed':42,'n':10,'m':1000,'original':data.tolist(),
                  'first_resample_ids_1based':(ids[:3]+1).tolist(),'first_means':means[:3].tolist()},
    'boxplot': {'data':values.tolist(),'q1':q1,'median':med,'q3':q3,'iqr':iqr,
                'fences':[low,high],'whiskers':[stats['whislo'],stats['whishi']],
                'fliers':stats['fliers'].tolist()},
    'banner': {'original_dimensions':[1499,949],'new_html_frame_ratio':[2998,949],
               'height_ratio_at_equal_width':0.5}
}, ensure_ascii=False, indent=2),encoding='utf8')
print('Created four scientific figures and verified boxplot statistics.')
