
import numpy as np
import pandas as pd

from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests

from utils.plots import *
import seaborn as sns

def cmap(name):
    if 'relaxed' in name.lower():
        return cp[3]
    if 'ablated' in name.lower() or 'uninformative' in name.lower():
        return cp[0]
    if name.startswith('Bayes') or 'prior' in name or 'uninformative' in name:
        return cp[1]
    if name.startswith('Ideal'):
        return cp[-1]
    else:
        return cp[3]


def plot_metrics(df_metrics, metric, methods, names,
                 title=None, xlabel=None, ylabel=None, log=False, seed=0,
                 markersize=2, boxwidth=1):

    if log:
        plt.semilogy()

    hues = [cmap(name) for name in names]

    df_temp = df_metrics.copy()
    df_temp = df_temp[df_temp['method'].isin(methods)]
    is_keep = df_temp.groupby('i')[metric].agg(lambda x: not np.any(pd.isna(x)))
    keepers = is_keep.index[is_keep]
    df_temp = df_temp[df_temp['i'].isin(keepers)]

    np.random.seed(seed)
    sns.stripplot(df_temp, x='method', y=metric, size=markersize, jitter=0.25*boxwidth, 
                  order=methods, hue='method', palette=hues, hue_order=methods, zorder=-1)
    sns.boxplot(df_temp, x='method', y=metric, order=methods,
                fill=None, color='#303030', width=0.8*boxwidth, fliersize=0)
    sns.despine(ax=plt.gca(), bottom=True)
    plt.xticks(range(len(methods)), names)
    plt.gca().tick_params(length=0, rotation=90, axis='x')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)

    return df_temp


def print_stats(df, metric, methods, names, bonferroni=True):

    print('median metrics:')
    print(df.groupby('method')[metric].agg('median'))
    print()
    
    pvals = []
    stats = []
    name_pairs = []
    for i in range(len(methods)-1):
        for j in range(i+1,len(methods)):

            x1 = df.query(f'method=="{methods[i]}"')[metric].values
            x2 = df.query(f'method=="{methods[j]}"')[metric].values

            result = wilcoxon(x1, x2, alternative='two-sided', nan_policy='omit')
            pvals.append(result.pvalue)
            stats.append(result.statistic)
            name_pairs.append((names[i], names[j]))

    if bonferroni:
        result = multipletests(pvals, alpha=0.05, method='bonferroni')
        pvals = result[1]
    
    for pval, stat, (name1, name2) in zip(pvals, stats, name_pairs):
        if pval < 0.001:
            stars = '***'
        elif pval < 0.01:
            stars = '**'
        elif pval < 0.05:
            stars = '*'
        else:
            stars = ''

        if stars:
            sign = '<>'
        else:
            sign = '=='
        
        print(stars, end='\t')
        print(f'p={pval:0.3E}, {name1} {sign} {name2}')
    print()
