import numpy as np
import copy
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from pyrepo_mcda import normalizations as norms
from pyrepo_mcda.additions import rank_preferences
from vassp import VASSP

from pyrepo_mcda import weighting_methods as mcda_weights
from pyrepo_mcda.mcda_methods import TOPSIS, VIKOR, COPRAS
from pyrepo_mcda import correlations as corrs



# bar (column) chart
def plot_barplot(df_plot, legend_title, comment = ''):
    """
    Visualization method to display column chart of alternatives rankings obtained with 
    different methods.
    Parameters
    ----------
        df_plot : DataFrame
            DataFrame containing rankings of alternatives obtained with different methods.
            The particular rankings are included in subsequent columns of DataFrame.
        title : str
            Title of the legend (Name of group of explored methods, for example MCDA methods or Distance metrics).
    
    Examples
    ----------
    >>> plot_barplot(df_plot, legend_title='MCDA methods')
    """
    step = 1
    list_rank = np.arange(1, len(df_plot) + 1, step)

    ax = df_plot.plot(kind='bar', width = 0.8, stacked=False, edgecolor = 'black', figsize = (10,5))
    ax.set_xlabel('Alternatives', fontsize = 12)
    ax.set_ylabel('Rank', fontsize = 12)
    ax.set_yticks(list_rank)

    ax.set_xticklabels(df_plot.index, rotation = 'horizontal')
    ax.tick_params(axis = 'both', labelsize = 12)
    y_ticks = ax.yaxis.get_major_ticks()
    ax.set_ylim(0, len(df_plot) + 1)

    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',
    ncol=4, mode="expand", borderaxespad=0., edgecolor = 'black', title = legend_title, fontsize = 12)

    ax.grid(True, linestyle = '--')
    ax.set_axisbelow(True)
    plt.tight_layout()
    legend_title = legend_title.replace("$", "")
    legend_title = legend_title.replace(" ", "_")
    plt.savefig('./results/' + 'bar_chart_' + legend_title + comment + '.png')
    plt.savefig('./results/' + 'bar_chart_' + legend_title + comment + '.eps')
    plt.savefig('./results/' + 'bar_chart_' + legend_title + comment + '.pdf')
    plt.show()


# radar chart
def plot_radar(data, title, comment = ''):
    """
    Visualization method to display rankings of alternatives obtained with different methods
    on the radar chart.

    Parameters
    -----------
        data : DataFrame
            DataFrame containing containing rankings of alternatives obtained with different 
            methods. The particular rankings are contained in subsequent columns of DataFrame.
        title : str
            Chart title

    Examples
    ----------
    >>> plot_radar(data, title)
    """
    fig=plt.figure()
    ax = fig.add_subplot(111, polar = True)

    for col in list(data.columns):
        labels=np.array(list(data.index))
        stats = data.loc[labels, col].values

        angles=np.linspace(0, 2*np.pi, len(labels), endpoint=False)
        # close the plot
        stats=np.concatenate((stats,[stats[0]]))
        angles=np.concatenate((angles,[angles[0]]))
    
        lista = list(data.index)
        lista.append(data.index[0])
        labels=np.array(lista)

        ax.plot(angles, stats, '-o', linewidth=2)
    
    ax.set_thetagrids(angles * 180/np.pi, labels)
    ax.set_rgrids(np.arange(1, data.shape[0] + 1, 2))
    ax.grid(True, linestyle = '--')
    ax.set_axisbelow(True)
    plt.legend(data.columns, bbox_to_anchor=(1.0, 0.95, 0.4, 0.2), loc='upper left')
    plt.title(title)
    plt.tight_layout()
    plt.savefig('./results/' + 'radar_chart' + comment + '.eps')
    plt.savefig('./results/' + 'radar_chart' + comment + '.png')
    plt.savefig('./results/' + 'radar_chart' + comment + '.pdf')
    plt.show()


# heat maps with correlations
def draw_heatmap(df_new_heatmap, title, kind = ''):
    """
    Visualization method to display heatmap with correlations of compared rankings generated using different methods
    
    Parameters
    ----------
        data : DataFrame
            DataFrame with correlation values between compared rankings
        title : str
            title of chart containing name of used correlation coefficient
    
    Examples
    ---------
    >>> draw_heatmap(df_new_heatmap, title)
    """
    plt.figure(figsize = (8, 6))
    
    sns.set(font_scale = 1.4)
    heatmap = sns.heatmap(df_new_heatmap, annot=True, fmt=".4f", cmap="RdYlGn",
                          linewidth=0.5, linecolor='w')
    plt.yticks(va="center")
    plt.xlabel('MCDA methods')
    plt.ylabel('MCDA methods')
    plt.title('Correlation coefficient: ' + title)
    plt.tight_layout()
    title = title.replace("$", "")
    plt.savefig('./results/' + 'correlations_' + title + '_' + kind + '.eps')
    plt.savefig('./results/' + 'correlations_' + title + '_' + kind + '.png')
    plt.savefig('./results/' + 'correlations_' + title + '_' + kind + '.pdf')
    plt.show()


# heat maps with correlations
def draw_heatmap2(df_new_heatmap, title, kind = ''):
    """
    Visualization method to display heatmap with correlations of compared rankings generated using different methods
    
    Parameters
    ----------
        data : DataFrame
            DataFrame with correlation values between compared rankings
        title : str
            title of chart containing name of used correlation coefficient
    
    Examples
    ---------
    >>> draw_heatmap(df_new_heatmap, title)
    """
    plt.figure(figsize = (11, 5))
    
    sns.set(font_scale = 1.1)
    heatmap = sns.heatmap(df_new_heatmap, annot=True, fmt=".4f", cmap="RdYlGn",
                          linewidth=0.5, linecolor='w')
    plt.yticks(va="center")
    plt.xlabel(r'$s$' + ' coefficient for SSP-VIKOR')
    plt.ylabel('Classical MCDA methods')
    plt.title('Correlation coefficient: ' + title)
    plt.tight_layout()
    title = title.replace("$", "")
    plt.savefig('./results/' + 'correlations_' + title + '_' + kind + '.eps')
    plt.savefig('./results/' + 'correlations_' + title + '_' + kind + '.png')
    plt.savefig('./results/' + 'correlations_' + title + '_' + kind + '.pdf')
    plt.show()


# Create dictionary class
class Create_dictionary(dict):
  
    # __init__ function
    def __init__(self):
        self = dict()
          
    # Function to add key:value
    def add(self, key, value):
        self[key] = value




def main():

    # Load Symbols of Countries
    coun_names = pd.read_csv('./data/country_names.csv')
    country_names = list(coun_names['Symbol'])

    # Choose evaluated year: 2020 or 2021
    year = '2021'
    df_data = pd.read_csv('./data/data_' + year + '.csv', index_col='Country')
    # Criteria types
    types = np.array([1, 1, 1, 1, 1, 1, -1, -1, 1, 1, -1, -1, -1])
    matrix = df_data.to_numpy()
    # Determine criteria weights
    weights = mcda_weights.critic_weighting(matrix)
    
    # Initialize the SSP-VIKOR method
    vassp = VASSP()

    df_results_rank = pd.DataFrame(index = country_names)
    df_results_pref = pd.DataFrame(index = country_names)

    # 4 strategies of sustainability coefficient s values (compensation reduction)
    # 1 - Scenario: sustainability coefficient = 0.3
    s_tri = np.ones(matrix.shape[1]) * 0.3
    pref_vassp = vassp(matrix, weights, types, s_coeff = s_tri)
    rank_vassp = rank_preferences(pref_vassp, reverse = False)


    df_results_rank['0.3'] = rank_vassp
    df_results_pref['0.3'] = pref_vassp

    df_results_pref.to_csv('./results/results_pref_03_' + year + '.csv')
    df_results_rank.to_csv('./results/results_rank_03_' + year + '.csv')


    # 2 - standard deviation (std)
    n_matrix = norms.minmax_normalization(matrix, types)
    s_std = np.sqrt(np.sum(np.square(np.mean(n_matrix, axis = 0) - n_matrix), axis = 0) / n_matrix.shape[0])

    pref_vassp = vassp(matrix, weights, types, s_coeff = s_std)
    rank_vassp = rank_preferences(pref_vassp, reverse = False)

    df_results_rank['std'] = rank_vassp
    df_results_pref['std'] = pref_vassp

    # 3 - reduced compensation s = 1.0 for each criterion
    s_ones = np.ones(matrix.shape[1])
    pref_vassp = vassp(matrix, weights, types, s_coeff = s_ones)
    rank_vassp = rank_preferences(pref_vassp, reverse = False)

    df_results_rank['1.0'] = rank_vassp
    df_results_pref['1.0'] = pref_vassp

    # 4 - full compensation s = 0.0 for each criterion
    s_zeros = np.zeros(matrix.shape[1])
    pref_vassp = vassp(matrix, weights, types, s_coeff = s_zeros)
    rank_vassp = rank_preferences(pref_vassp, reverse = False)

    df_results_rank['0.0'] = rank_vassp
    df_results_pref['0.0'] = pref_vassp


    df_results_pref = df_results_pref.rename_axis('Country')
    df_results_rank = df_results_rank.rename_axis('Country')
    
    df_results_pref.to_csv('./results/results_pref_1_' + year + '.csv')
    df_results_rank.to_csv('./results/results_rank_1_' + year + '.csv')


    # simulation with changing of s coefficient values
    std = np.mean(s_std)
    vec_sust = np.array([0.0, 0.25, std, 0.5, 0.75, 1.0])

    
    df2rank = pd.DataFrame(index = country_names)
    df2pref = pd.DataFrame(index = country_names)

    for vs in vec_sust:
        s = np.ones(matrix.shape[1]) * vs
        pref_vassp = vassp(matrix, weights, types, s_coeff = s)
        rank_vassp = rank_preferences(pref_vassp, reverse = False)

        df2rank["{:.2f}".format(vs)] = rank_vassp
        df2pref["{:.2f}".format(vs)] = pref_vassp


    df2pref = df2pref.rename_axis('Country')
    df2rank = df2rank.rename_axis('Country')


    df2pref.to_csv('./results/results_pref_2_' + year + '.csv')
    df2rank.to_csv('./results/results_rank_2_' + year + '.csv')


    # plot figure with sensitivity analysis

    for k in range(df2rank.shape[0]):
        plt.plot(vec_sust, df2rank.iloc[k, :])

        ax = plt.gca()
        y_min, y_max = ax.get_ylim()
        x_min, x_max = ax.get_xlim()
        
        plt.annotate(country_names[k], (x_max, df2rank.iloc[k, -1]),
                        fontsize = 12, style='italic',
                        horizontalalignment='left')

    plt.xlabel(r'$s$' + ' coefficient')
    plt.ylabel("Rank", )
    
    os_x = np.linspace(0, 1, 6)
    
    vec_sust_std = [0.0, 0.25, r'$std$', 0.5, 0.75, 1.0]
    plt.xticks(os_x, vec_sust_std)
    plt.yticks(ticks=np.arange(1, len(country_names) + 1, 1))
    plt.gca().invert_yaxis()
    plt.grid(True, linestyle = '--')
    plt.tight_layout()
    plt.savefig('results/sust_coeff_' + year + '.png')
    plt.savefig('results/sust_coeff_' + year + '.eps')
    plt.savefig('results/sust_coeff_' + year + '.pdf')
    plt.show()


    # =====================================================================================
    # PART 2 - Experiments
    # --------------------------------------------------------------------------------------
    # Study 1 comparative analysis
    df3rank = pd.DataFrame(index = country_names)
    
    # VASSP
    s_tri = np.ones(matrix.shape[1]) * 0.3
    pref_vassp = vassp(matrix, weights, types, s_coeff = s_tri)
    rank_vassp = rank_preferences(pref_vassp, reverse = False)
    df3rank['SSP-VIKOR'] = rank_vassp

    # VIKOR
    vikor = VIKOR()
    pref = vikor(matrix, weights, types)
    rank_vik = rank_preferences(pref, reverse = False)
    df3rank['VIKOR'] = rank_vik
    # TOPSIS
    topsis = TOPSIS()
    pref = topsis(matrix, weights, types)
    rank_tops = rank_preferences(pref, reverse = True)
    df3rank['TOPSIS'] = rank_tops
    # COPRAS
    copras = COPRAS()
    pref = copras(matrix, weights, types)
    rank_cop = rank_preferences(pref, reverse = True)
    df3rank['COPRAS'] = rank_cop

    # Rankings correlations
    results = copy.deepcopy(df3rank)
    method_types = list(results.columns)
    dict_new_heatmap_rw = Create_dictionary()

    for el in method_types:
        dict_new_heatmap_rw.add(el, [])


    # heatmaps for correlations coefficients
    for i, j in [(i, j) for i in method_types[::-1] for j in method_types]:
        dict_new_heatmap_rw[j].append(corrs.weighted_spearman(results[i], results[j]))

    df_new_heatmap_rw = pd.DataFrame(dict_new_heatmap_rw, index = method_types[::-1])
    df_new_heatmap_rw.columns = method_types

    # correlation matrix with rw coefficient
    draw_heatmap(df_new_heatmap_rw, r'$r_w$', 'bad1')

    # -------------------------------------------------------------------------------------
    # Study 2 comparative analysis
    s_coeffs = np.arange(0.0, 1.1, 0.1)

    dict_bad2_rw = Create_dictionary()
    list_cols = ["{:.1f}".format(el) for el in s_coeffs]

    for el in list_cols:
        dict_bad2_rw.add(el, [])


    for el, vs in enumerate(s_coeffs):
        s = np.ones(matrix.shape[1]) * vs
        pref_vassp = vassp(matrix, weights, types, s_coeff = s)
        rank_vassp = rank_preferences(pref_vassp, reverse = False)
        dict_bad2_rw[list_cols[el]].append(corrs.weighted_spearman(rank_vassp, rank_vik))
        dict_bad2_rw[list_cols[el]].append(corrs.weighted_spearman(rank_vassp, rank_tops))
        dict_bad2_rw[list_cols[el]].append(corrs.weighted_spearman(rank_vassp, rank_cop))


    df_bad2_rw = pd.DataFrame(dict_bad2_rw, index = ['VIKOR', 'TOPSIS', 'COPRAS'])
    print(df_bad2_rw)
    draw_heatmap2(df_bad2_rw, r'$r_w$', 'bad2')

    

        

        

        
        







    
    

if __name__ == '__main__':
    main()