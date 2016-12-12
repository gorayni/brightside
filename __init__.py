from jinja2 import Environment, FileSystemLoader
from IPython.display import display

import inspect
import os

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


def print_attribute(tag, value):
    display(Attribute(tag, value))


def print_bold(value):
    display(Bold(value))
