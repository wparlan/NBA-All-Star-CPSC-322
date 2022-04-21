import matplotlib.pyplot as plt
import utils
import numpy as np

def bar_chart(x, y, title, x_label, y_label, tilt=False,grouped=False):
    plt.figure(figsize=(14, 5))
    if grouped:
        plt.bar(x, y,width=1, edgecolor='black')
    else:
        plt.bar(x, y)
    if tilt:
        plt.xticks(rotation=30, ha="right")

    plt.grid(axis='y', linestyle='--')
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.show()

def histogram(bins_list, labels, title, x_label, y_label, edge=True):
    # data is a 1D list of data values
    plt.figure()
    if len(labels) and edge:
        for bin, label in zip(bins_list, labels):
            plt.hist(bin, label=label,edgecolor='black') # default is 10
            plt.legend()
    elif edge:
        for bin in bins_list:
            plt.hist(bin, edgecolor='black') # default is 
    else:
        for bin in bins_list:
            plt.hist(bin) # default is 

    plt.title(title)
    plt.grid(axis='y', linestyle='--')
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.show()

def multi_bar_chart(category_names, heights, bar_names, title, x_label, y_label):
    # X = ['Group A','Group B','Group C','Group D']
    # Ygirls = [10,20,20,40]
    # Zboys = [20,30,25,30]
    
    # X_axis = np.arange(len(X))
    x_axis = range(len(category_names))
    print(x_axis)
    # plt.bar(X_axis - 0.2, Ygirls, 0.4, label = 'Girls')
    spacing = 0.2
    for bar, name in zip(heights, bar_names):
        plt.bar(x_axis - spacing, bar, 0.4, label=name)
        if spacing > 0:
            spacing *= -1
        else:
            spacing = -spacing+0.2
    # plt.bar(X_axis + 0.2, Zboys, 0.4, label = 'Boys')
    
    # plt.xticks(X_axis, X)
    plt.xticks(x_axis, category_names)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.legend()
    plt.show()
    return

def pie_chart(x, labels, title):
    plt.figure(figsize=(14, 5))
    plt.pie(x, labels=labels, autopct="%1.1f%%")
    plt.axis('equal')
    plt.title(title)
    plt.show()

def scatter_chart(xs, ys, title, x_label, y_label, correlate=True, xlim=None, ylim=None):
    # plt.figure(figsize=(10, 10)) # init figure
    # plot data from xs, ys (parallel lists of parallel lists)
    for x, y in zip(xs, ys):
        plt.scatter(x, y)
        if correlate:
            m, b = utils.compute_slope_intercept(x, y)
            plt.plot([min(x), max(x)], [m * min(x) + b, m * max(x) + b], c="r", lw=5, label='Correlation')
            plt.legend()
            r = utils.compute_correlation(x, y)
            cov = utils.compute_covariance(x, y)
            plt.annotate(f"corr: {format(r, '.2f')}\ncov: {format(cov, '.2f')}", xy=(0.50, 0.88), xycoords="axes fraction", 
                        horizontalalignment="center", color="r", bbox=dict(boxstyle="round", fc="1", color="r"))
    # format chart
    plt.title(title)
    plt.grid(linestyle='--')
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    if xlim is not None:
        plt.xlim(xlim[0], xlim[1])
    if ylim is not None:
        plt.ylim(ylim[0], ylim[1])
    plt.show()

def box_plot(distributions, labels, title, x_label, y_label): # distributions and labels are parallel
    # distributions: list of 1D lists of values
    plt.figure(figsize=(14, 5))
    plt.boxplot(distributions)
    # boxes correspond to the 1st and 3rd quartiles
    # line in the middle of the box corresponds to the 2nd quartile (AKA median)
    # whiskers corresponds to +/- 1.5 * IQR
    # IQR: interquartile range (3rd quartile - 1st quartile)
    # circles outside the whiskers correspond to outliers

    # customize x ticks
    plt.xticks(list(range(1, len(distributions) + 1)), labels)


    # annotations
    # we want to add "mu=100" to the center of our figure
    # xycoords="data": default, specify the location of the label in the same
    # xycoords = "axes fraction": specify the location of the label in absolute
    # axes coordinates... 0,0 is the lower left corner, 1,1 is the upper right corner
    # coordinates as the plotted data
    # plt.annotate("$\mu=100$", xy=(1.5, 105), xycoords="data", horizontalalignment="center")
    # plt.annotate("$\mu=100$", xy=(0.5, 0.5), xycoords="axes fraction", 
    #              horizontalalignment="center", color="blue")

    plt.grid(axis='y', linestyle='--')
    plt.xticks(rotation=30, ha="right")
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.show()