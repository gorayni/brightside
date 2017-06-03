from __future__ import division

import inspect
import os
import types
import warnings

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import sklearn.metrics as metrics
from IPython.display import display
from jinja2 import Environment, FileSystemLoader
from matplotlib.colors import ListedColormap
from pylab import annotate

import pandas as pd
from pandas import Series

PATH = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
TEMPLATE_ENVIRONMENT = Environment(autoescape=False, loader=FileSystemLoader(PATH), trim_blocks=False)


class Attribute():
    def __init__(self, attribute, value):
        self.html = '<b>' + str(attribute) + ' </b>' + str(value)
        self.latex = '\\textbf{' + str(attribute) + '} ' + str(value)

    def _repr_html_(self):
        return self.html

    def _repr_latex_(self):
        return self.latex


class LabelValueTable():
    def __init__(self, title, labels_and_values):
        html_template = TEMPLATE_ENVIRONMENT.get_template('label_value_table_template.html')
        latex_template = TEMPLATE_ENVIRONMENT.get_template('label_value_table_template.tex')

        self.html = html_template.render(title=title, labels_and_values=labels_and_values)
        self.latex = latex_template.render(title=title, labels_and_values=labels_and_values)

    def _repr_html_(self):
        return self.html

    def _repr_latex_(self):
        return self.latex


class Bold():
    def __init__(self, value):
        self.html = '<b>' + str(value) + ' </b>'
        self.latex = '\\textbf{' + str(value) + '} '

    def _repr_html_(self):
        return self.html

    def _repr_latex_(self):
        return self.latex


def map_new_categories(categories, groundtruth, predictions, labels, other_label='other'):
    categories = sorted(list(categories))

    # Creating new categories vectorization function
    new_categories = {cat: i for i, cat in enumerate(categories)}
    other_category = len(new_categories)

    def map_function(x):
        return new_categories.get(x, other_category)

    to_new_categories = np.vectorize(map_function)

    new_groundtruth = to_new_categories(groundtruth)
    new_predictions = to_new_categories(predictions)
    new_labels = [labels[i] for i in categories]
    new_labels.append(other_label)

    return new_groundtruth, new_predictions, new_labels


def show_confusion_matrix(true_labels, predicted_labels, labels, figsize=None, normalize=True, annot=False, cmap='jet',
                          ticks_size=None, linewidths=0, show_yticks=True, show_xticks=False):
    if not figsize:
        figsize = (5, 5)

    cm = metrics.confusion_matrix(true_labels, predicted_labels)
    # normalize confusion matrix
    if normalize:
        num_instances_per_class = cm.sum(axis=1)
        zero_indices = num_instances_per_class == 0
        if any(zero_indices):
            num_instances_per_class[zero_indices] = 1
            warnings.warn('One or more classes does not have instances')
        cm = cm / num_instances_per_class[:, np.newaxis]
        vmax = 1.
    else:
        vmax = np.max(cm)

    fig = plt.figure(figsize=figsize)
    plt.clf()
    ax = fig.add_subplot(111)
    ax.set_aspect(1)

    if show_yticks:
        yticklabels = labels
    else:
        yticklabels = []

    if show_xticks:
        xticklabels = labels
    else:
        xticklabels = []

    ax = sns.heatmap(cm, annot=annot, cmap=cmap, linewidths=linewidths, xticklabels=xticklabels,
                     yticklabels=yticklabels, vmax=vmax)

    if show_yticks:
        for ticklabel in ax.get_yaxis().get_ticklabels():
            ticklabel.set_rotation('horizontal')
            if ticks_size:
                ticklabel.set_fontsize(ticks_size)

    if show_xticks:
        ax.xaxis.tick_top()
        for ticklabel in ax.get_xaxis().get_ticklabels():
            ticklabel.set_rotation('vertical')
            if ticks_size:
                ticklabel.set_fontsize(ticks_size)

    return fig, ax


def show_sequences(sequences, labels_colors=None, figsize=None, tight_layout=None, mask_value=None, ylabel=None,
                   xlabel=None, yticklabels=True, xticklabels=True, leg_square_size=10, annot=False, aspect_ratio=None,
                   show_box=False, plot_ylabel=None, plot_xlabel=None, title=None, sequence_ind=None):
    fig = plt.figure(figsize=figsize)
    plt.clf()
    ax = fig.add_subplot(111)

    if title:
        plt.title(title)

    if aspect_ratio:
        ax.set_aspect(aspect_ratio)

    if not sequence_ind:
        if isinstance(sequences, pd.DataFrame):
            sequence_ind = Series(sequences.values.ravel()).unique()
            sequence_ind = np.sort(sequence_ind)
        else:
            sequence_ind = np.unique(sequences)

    # Removing mask value from unique sequences
    if mask_value:
        mask_ind = np.argwhere(sequence_ind == mask_value)
        if len(mask_ind) > 0:
            sequence_ind = np.delete(sequence_ind, mask_ind)

    if not labels_colors:
        colors = sns.color_palette("hls", len(sequence_ind))
        labels_colors = {seq_ind: ("", colors[i]) for i, seq_ind in enumerate(sequence_ind)}

        vmax = sequence_ind[-1]
        vmin = sequence_ind[0]
    else:
        vmax = np.max(labels_colors.keys())
        vmin = np.min(labels_colors.keys())

    cmap = ListedColormap([color for k, (label, color) in labels_colors.iteritems()])

    sns.set_style("white", {'grid.color': '.9', 'axes.edgecolor': '.2', 'axes.linewidth': 1})

    if xticklabels:
        if not isinstance(sequences, pd.DataFrame):
            pass
        if not type(xticklabels):
            xticklabels = np.linspace(0, sequences.shape[1], xticklabels)

    if mask_value:
        ax = sns.heatmap(sequences,
                         cmap=cmap, cbar=False,
                         annot=annot,
                         vmin=vmin, vmax=vmax,
                         xticklabels=xticklabels,
                         yticklabels=yticklabels,
                         mask=sequences == mask_value)
    else:
        ax = sns.heatmap(sequences,
                         cmap=cmap, cbar=False,
                         annot=annot,
                         vmin=vmin, vmax=vmax,
                         xticklabels=xticklabels,
                         yticklabels=yticklabels)

    if show_box:
        ax.axhline(y=0, color='k', linewidth=2)
        ax.axhline(y=sequences.shape[0], color='k', linewidth=2)
        ax.axvline(x=0, color='k', linewidth=2)
        ax.axvline(x=sequences.shape[1], color='k', linewidth=2)

    if plot_ylabel:
        plt.ylabel(plot_ylabel, fontweight='bold', fontsize=12)
    if plot_xlabel:
        plt.xlabel(plot_xlabel, fontweight='bold', fontsize=12)

    if ylabel:
        ax.set_ylabel(ylabel)
    if xlabel:
        ax.set_xlabel(xlabel)

    def create_proxy(color):
        line = matplotlib.lines.Line2D([0], [0], linestyle='none', mfc=color,
                                       mec='black',
                                       markersize=leg_square_size,
                                       marker='s', )
        return line

    proxies = [create_proxy(labels_colors[ind][1]) for ind in sequence_ind]

    if annot:
        descriptions = ['{} {}'.format(ind, labels_colors[ind][0]) for ind in sequence_ind]
    else:
        descriptions = ['{}'.format(labels_colors[ind][0]) for ind in sequence_ind]

    leg = ax.legend(proxies, descriptions, numpoints=1, markerscale=2, frameon=True,
                    bbox_transform=plt.gcf().transFigure, bbox_to_anchor=(1, 0.5), loc=10)
    leg.get_frame().set_edgecolor('#000000')
    leg.get_frame().set_linewidth(0)
    leg.get_frame().set_facecolor('#FFFFFF')
    if tight_layout:
        plt.tight_layout()

    return fig, ax


def plot_datasets_summary(stats, figsize=None, ylabel="Number of Instances", xlabel="Categories", annot_rotation=90,
                          annot_fontsize=9, axis_fontsize=12, legend=False, width=0.5, annotate_cols=True, title=None,
                          colormap=None):
    if not figsize:
        figsize = (1, 1)
    sns.set_style("whitegrid", {'grid.color': '.9', 'axes.edgecolor': '.2', 'axes.linewidth': 1})
    sns.set_context("notebook", font_scale=1, rc={"lines.linewidth": 1})

    fig, ax = plt.subplots(1, 1, figsize=figsize, sharex=True)

    if title:
        plt.title(title)

    if stats.shape[1] == 1:
        stats.plot(kind='bar', ax=ax, legend=legend, width=width, color=[sns.color_palette("PuBu", 10)[-3]])
    else:
        stats.plot(kind='bar', ax=ax, legend=legend, width=width, colormap=colormap)

    plt.setp(ax.xaxis.get_majorticklabels(), rotation=annot_rotation, fontsize=axis_fontsize, ha='right')
    plt.setp(ax.yaxis.get_majorticklabels(), fontsize=axis_fontsize)

    if annotate_cols:
        for i, (index, row) in enumerate(stats.iterrows()):
            for col in stats.columns.values:
                annotate('{:.0f}'.format(row[col]), (i, row[col] + 25), ha='center', fontsize=annot_fontsize)
        plt.ylim([0, 1.1 * stats.values.max()])

    SHIFT = -0.3  # Data coordinates
    for label in ax.xaxis.get_majorticklabels():
        label.customShiftValue = SHIFT
        label.set_x = types.MethodType(lambda self, x: matplotlib.text.Text.set_x(self, x - self.customShiftValue),
                                       label, matplotlib.text.Text)

    plt.ylabel(ylabel, fontweight='bold', fontsize=12)
    plt.xlabel(xlabel, fontweight='bold', fontsize=12)
    return fig, ax

def plot_results(values, labels=None, iters=None, epochs=None, figsize=None, plot_type='accuracy'):
    
    if not isinstance(values, list):
        values = [values]

    if iters is not None:
        x_value = iters
        label = u'Iteration'
    elif epochs is not None:
        x_value = epochs
        label = u'Epoch'
    else:        
        x_value = np.arange(1, len(values[0]) + 1)
        label = u'Time'

    if not figsize:
        figsize = (1, 1)

    sns.set_style("whitegrid", {'axes.edgecolor': '.1', 'axes.linewidth': 0.8})
    sns.set_palette("muted")
    sns.set_context("notebook", font_scale=1, rc={"lines.linewidth": 1})

    fig, ax = plt.subplots(1, 1, figsize=figsize, sharex=True)
    plt.grid(b=True, which='major', color='#555555', linestyle=':', linewidth=0.5)

    for i, v in enumerate(values):
        if labels:
            plt.plot(x_value, v, linewidth=1.25, linestyle='-', marker='o', markersize=4, label=labels[i])
        else:
            plt.plot(x_value, v, linewidth=1.25, linestyle='-', marker='o', markersize=4)

    if plot_type == 'accuracy':
        plt.xlim(x_value[0], x_value[-1])
        plt.ylim(0, 1)
        plt.yticks(np.linspace(0, 1, 11, endpoint=True))

        plt.ylabel(u'Accuracy', fontweight='bold', fontsize=12)
        plt.xlabel(label, fontweight='bold', fontsize=12)
        plt.legend(loc=4, fontsize=11, frameon=True)

        axes = plt.gca()
        axes.set_aspect(x_value[-1])

    elif plot_type == 'loss':
        plt.xlim(x_value[0], x_value[-1])

        plt.ylabel(u'Error', fontweight='bold', fontsize=12)
        plt.xlabel(label, fontweight='bold', fontsize=12)
        plt.legend(loc=1, fontsize=11, frameon=True)
        axes = plt.gca()
        axes.set_aspect(500)
    

    return fig, ax


def plot_accuracy(values, labels=None, iters=None, epochs=None, figsize=None):
    return plot_results(values, labels, iters, epochs, figsize, plot_type='accuracy')


def plot_loss(values, labels=None, iters=None, epochs=None, figsize=None):
    return plot_results(values, labels, iters, epochs, figsize, plot_type='loss')


def print_attribute(tag, value):
    display(Attribute(tag, value))


def print_bold(value):
    display(Bold(value))
