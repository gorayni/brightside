import inspect
import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import sklearn.metrics as metrics
from IPython.display import display
from jinja2 import Environment, FileSystemLoader

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
        cm = cm / num_instances_per_class[:, np.newaxis]

    fig = plt.figure(figsize=figsize)
    plt.clf()
    ax = fig.add_subplot(111)
    ax.set_aspect(1)

    sns.heatmap(cm, annot=annot, cmap=cmap, linewidths=linewidths).get_figure()

    num_labels = len(labels)

    plt.yticks(range(num_labels), labels[::-1])
    plt.xticks([], [])
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(label_size)
    return fig, ax


def print_attribute(tag, value):
    display(Attribute(tag, value))


def print_bold(value):
    display(Bold(value))
