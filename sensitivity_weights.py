import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pyrepo_mcda.additions import rank_preferences
from vassp import VASSP

def prepare_weights(g_ind, val_mod, n = 13):
    
    rest = 1 - val_mod
    rrest = rest / 6
    weights = np.ones(n) * (rrest / 2)

    if g_ind == 0:
        weights[0] = val_mod / 2
        weights[1] = val_mod / 2
        weights[12] = rrest
    elif g_ind == 1:
        weights[2] = val_mod / 2
        weights[3] = val_mod / 2
        weights[12] = rrest
    elif g_ind == 2:
        weights[4] = val_mod / 2
        weights[5] = val_mod / 2
        weights[12] = rrest
    elif g_ind == 3:
        weights[6] = val_mod / 2
        weights[7] = val_mod / 2
        weights[12] = rrest
    elif g_ind == 4:
        weights[8] = val_mod / 2
        weights[9] = val_mod / 2
        weights[12] = rrest
    elif g_ind == 5:
        weights[10] = val_mod / 2
        weights[11] = val_mod / 2
        weights[12] = rrest
    elif g_ind == 6:
        weights[12] = val_mod

    return weights


def main():
    # Symbols of Countries
    coun_names = pd.read_csv('./data/country_names.csv')
    country_names = list(coun_names['Symbol'])

    # choose evaluated year: 2020 or 2021
    year = '2021'
    df_data = pd.read_csv('./data/data_' + year + '.csv', index_col='Country')
    types = np.array([1, 1, 1, 1, 1, 1, -1, -1, 1, 1, -1, -1, -1])
    matrix = df_data.to_numpy()

    s_tri = np.ones(matrix.shape[1]) * 0.3

    names = country_names

    indexes = np.array([0, 1, 2, 3, 4, 5, 6])
    w_values = np.arange(0.1, 0.8, 0.1)
    vassp = VASSP()

    # G1, G2, G3, ...
    for ind in indexes:
        results_sa = pd.DataFrame(index = country_names)
        pref_sa = pd.DataFrame(index = country_names)
        # weights 0.1, 0.2, 0.3, ...
        for val_mod in w_values:
            weights = prepare_weights(ind, val_mod, n = 13)

            pref_vassp = vassp(matrix, weights, types, s_coeff = s_tri)
            rank_vassp = rank_preferences(pref_vassp, reverse = False)

            pref_sa[str(val_mod)] = pref_vassp
            results_sa[str(val_mod)] = rank_vassp
            
        
        results_sa.to_csv('./sensitivity/results_sa_G' + str(ind + 1) + '.csv')
        
        # plot results of analysis with criteria weights modification
        ticks = np.arange(1, 27)

        x1 = np.arange(0, len(w_values))

        plt.figure(figsize = (12, 6))
        for i in range(results_sa.shape[0]):
            plt.plot(x1, results_sa.iloc[i, :], 'o-', linewidth = 2)
            ax = plt.gca()
            y_min, y_max = ax.get_ylim()
            x_min, x_max = ax.get_xlim()
            plt.annotate(names[i], (x_max, results_sa.iloc[i, -1]),
                            fontsize = 12, #style='italic',
                            horizontalalignment='left')

        
        plt.xlabel('Modification of ' + r'$G_{' + str(ind + 1) + '}$' + ' weight', fontsize = 12)
        plt.ylabel("Rank", fontsize = 12)
        plt.xticks(x1, np.round(w_values, 2), fontsize = 12)
        plt.yticks(ticks, fontsize = 12)
        plt.xlim(x_min, x_max)
        plt.gca().invert_yaxis()

        plt.grid(True, linestyle = ':')
        plt.tight_layout()
        
        plt.savefig('./sensitivity/sensitivity_G' + str(ind + 1) + '.eps')
        plt.savefig('./sensitivity/sensitivity_G' + str(ind + 1) + '.png')
        plt.savefig('./sensitivity/sensitivity_G' + str(ind + 1) + '.pdf')
        plt.show()


if __name__ == '__main__':
    main()