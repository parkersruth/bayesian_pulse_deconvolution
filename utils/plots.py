
import matplotlib.pyplot as plt
from cycler import cycler

# color palette
cp = ['#39436F', '#A12E4A', '#ABAF31', '#31AEB5', '#C0C0C0']

mpl_style_main = {
    'figure.dpi': 72,
    'font.family': 'sans-serif',
    'font.sans-serif': ['Open Sans', 'Arial'],
    'axes.prop_cycle': cycler(color=cp[:-1]),
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.xmargin': 0,
    'figure.autolayout': True,
    'lines.linewidth': 1,
}
plt.rcParams.update(mpl_style_main)

