import numpy as np
import csv
import config
import matplotlib.pyplot as plt

from matplotlib.patches import Rectangle
from random import randint


def get_box_color(index=None):
    '''Retrieve plt-compatible color string based on object index'''
    colors = ['w', 'r', 'b', 'g', 'c', 'm', 'y', 'g', 'c', 'm', 'k']
    if index is None:
        return colors[randint(0,len(colors)-1)]
    return colors[index%len(colors)]

def get_box_rgbcolor(index=None):
    """Retrieve rgb color based on object index"""
    colors = [(0, 0, 0), (255, 0, 0), (0, 0, 255), (0, 255, 0), (128, 128, 0)]
    if index is None:
        return colors[randint(0, len(colors) - 1)]
    return colors[index % len(colors)]


def index2class(index=0):
    '''Convert index (int) to class name(string)'''
    classes = config.params['classes']
    return classes[index]

def class2index(class_= "background"):
    '''Convert class name tp index'''
    classes = config.params['classes']
    return classes.index(class_)

