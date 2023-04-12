import matplotlib.pyplot as plt

def plot_sets(history, set_names, metric_names, line_styles=['-', '--', '-.', ':'], cmp = plt.cm.get_cmap('tab10'), line_width=1, 
              figsize=(10, 10), title="Validation Plot", xlabel="Epoch", ylabel="Metric", legend_loc='best',save_fig = False, fig_name = 'plot.png'):

    fig, ax = plt.subplots(figsize=figsize)
    
    num_epochs = len(history[f'{metric_names[0]}'])

    for i, set_name in enumerate(set_names):
        for j, metric_name in enumerate(metric_names):
            index_name = f'{set_name}_{metric_name}' if set_name != 'train' else f'{metric_name}' 
            ax.plot(history[index_name], c=cmp(i), linestyle=line_styles[j], linewidth=line_width, label=set_name + ' ' + metric_name)

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(loc=legend_loc)

    if(save_fig):
        fig.savefig(fig_name)

    plt.show()

