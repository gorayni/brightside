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
from pylab import annotate

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


def show_confusion_matrix(true_labels, predicted_labels, labels, figsize=None, normalize=True, annot=False, cmap='jet',
                          label_size=10, linewidths=0):
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

    fig = plt.figure(figsize=figsize)
    plt.clf()
    ax = fig.add_subplot(111)
    ax.set_aspect(1)

    sns.heatmap(cm, annot=annot, cmap=cmap, linewidths=linewidths).get_figure()
    plt.yticks(xrange(len(labels)), labels[::-1])
    plt.xticks([], [])
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(label_size)
    return fig, ax


def plot_datasets_summary(stats, figsize=None, ylabel="Number of Instances", xlabel="Categories", annot_rotation=90,
                          annot_fontsize=9, axis_fontsize=12):
    if not figsize:
        figsize = (1, 1)

    sns.set_style("whitegrid", {'grid.color': '.9', 'axes.edgecolor': '.2', 'axes.linewidth': 1})
    sns.set_context("notebook", font_scale=1, rc={"lines.linewidth": 1})

    fig, ax = plt.subplots(1, 1, figsize=figsize, sharex=True)

    stats.plot(kind='bar', ax=ax, color=[sns.color_palette("PuBu", 10)[-3]], legend=False)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=annot_rotation, fontsize=axis_fontsize, ha='right')
    plt.setp(ax.yaxis.get_majorticklabels(), fontsize=axis_fontsize)

    images = stats['images'].tolist()
    for w, k in zip(images, range(len(images))):
        annotate("{:,d}".format(w), (k, w + 25), ha='center', fontsize=annot_fontsize)

    SHIFT = -0.3  # Data coordinates
    for label in ax.xaxis.get_majorticklabels():
        label.customShiftValue = SHIFT
        label.set_x = types.MethodType(lambda self, x: matplotlib.text.Text.set_x(self, x - self.customShiftValue),
                                       label, matplotlib.text.Text)

    plt.ylabel(ylabel, fontweight='bold', fontsize=12)
    plt.xlabel(xlabel, fontweight='bold', fontsize=12)
    return fig, ax


def print_attribute(tag, value):
    display(Attribute(tag, value))


def print_bold(value):
    display(Bold(value))
