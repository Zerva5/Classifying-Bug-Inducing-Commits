import matplotlib.pyplot as plt

def plot_supervised(logs, setNames, metricNames, lineStyles = ['-', '--', '.'], cmap = plt.colormaps.get_cmap('tab10'), 
linewidth=1, figsize=(12, 8), xlabel='Epoch', ylabel='Metric', title='Supervised Learning Metrics', xlim=None, ylim=None, save=False, savePath=None):
    """
    Plot the selected sets with the selected metrics on the same plot
    """

    fig, ax = plt.subplots(figsize=figsize)
    for s_index, setName in enumerate(setNames):
        styleIndex = 0
        for m_index, metricName in enumerate(metricNames):
            key = f'{setName}_{metricName}' if setName != "" else metricName
            ax.plot(logs[key], label="{} {}".format(setName, metricName), linewidth=linewidth, linestyle=lineStyles[m_index], color=cmap(s_index))

    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)

    ax.legend()
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    
    if save:
        fig.savefig(savePath)

    plt.show()

