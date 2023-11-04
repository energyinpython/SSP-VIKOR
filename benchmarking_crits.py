import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from pyrepo_mcda import weighting_methods as mcda_weights

from vassp import VASSP
from pyrepo_mcda.mcda_methods import TOPSIS, VIKOR, COPRAS

from pyrepo_mcda import correlations as corrs

from pyrepo_mcda.additions import rank_preferences
import seaborn as sns


def main():

    results_dict_rw = {
    'variable': [],
    'value': [],
    'Method': []
    }
    
    # Benchmarking simulations
    iterations = 1000
    crits = np.array([10, 14, 18, 22, 26, 30])
    crits_ax = np.arange(2, 14, 2)
    
    # Standard criteria number
    n = 13

    # Standard alternatives number
    m = 26

    x = []
    y = []
    w = []
    df_final = pd.DataFrame()
    

    vassp = VASSP()
    topsis = TOPSIS()
    vikor = VIKOR()
    copras = COPRAS()
    # main loop with number of iterations
    for i in range(iterations):
        

        for it, n in enumerate(crits):
            matrix = np.random.uniform(1, 1000, size = (m, n))
            weights = mcda_weights.critic_weighting(matrix)
            types = np.random.randint(0, 2, size = n)
            types[types == 0] = -1
            types[0] = 1
            types[1] = -1

            # VIKOR
            pref_v = vikor(matrix, weights, types)
            rank_v = rank_preferences(pref_v, reverse = False)
            # TOPSIS
            pref_t = topsis(matrix, weights, types)
            rank_t = rank_preferences(pref_t, reverse = True)
            # COPRAS
            pref_c = copras(matrix, weights, types)
            rank_c = rank_preferences(pref_c, reverse = True)

            # VASSP (SSP-VIKOR)
            s_coeff = np.ones(matrix.shape[1]) * 0.3
            pref_s = vassp(matrix, weights, types, s_coeff = s_coeff)
            rank_s = rank_preferences(pref_s, reverse = False)

            # correlation of SSP-VIKOR with VIKOR
            corr = corrs.weighted_spearman(rank_s, rank_v)

            results_dict_rw['variable'].append(n)
            results_dict_rw['value'].append(corr)
            results_dict_rw['Method'].append('VIKOR')

            y.append(corr)
            x.append(crits_ax[it] - 0.2)
            w.append('VIKOR')

            # correlation of SSP-VIKOR with TOPSIS
            corr = corrs.weighted_spearman(rank_s, rank_t)

            results_dict_rw['variable'].append(n)
            results_dict_rw['value'].append(corr)
            results_dict_rw['Method'].append('TOPSIS')

            y.append(corr)
            x.append(crits_ax[it])
            w.append('TOPSIS')

            # correlation of SSP-VIKOR with COPRAS
            corr = corrs.weighted_spearman(rank_s, rank_c)

            results_dict_rw['variable'].append(n)
            results_dict_rw['value'].append(corr)
            results_dict_rw['Method'].append('COPRAS')

            y.append(corr)
            x.append(crits_ax[it] + 0.2)
            w.append('COPRAS')


    # Criteria Alternatives
    df_final['Criteria'] = x
    df_final['Correlation'] = y
    df_final['Methods'] = w
    df_final = df_final.rename_axis('Lp')

    # plot visualization of benchmarking
    fig, ax = plt.subplots(figsize=(11, 7))
    ax = sns.scatterplot(x="Criteria", y="Correlation", data=df_final, hue="Methods", s = 50, alpha = 0.1, marker = 'o')
    ax.set_xticks(crits_ax)
    ax.set_xticklabels(crits, fontsize = 14)
    ax.set_xlabel('Criteria', fontsize = 14)
    ax.set_ylabel('Weighted Spearman correlation coefficient', fontsize = 14)
    
    plt.yticks(fontsize = 14)
    ax.set_title('')
    ax.grid(True, linestyle = '--')
    ax.set_axisbelow(True)
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',
    ncol=3, mode="expand", borderaxespad=0., fontsize = 12)
    plt.tight_layout()
    plt.savefig('benchmarkable/bench_correlationsCrits.png')
    plt.savefig('benchmarkable/bench_correlationsCrits.eps')
    plt.savefig('benchmarkable/bench_correlationsCrits.pdf')
    plt.show()



if __name__ == '__main__':
    main()