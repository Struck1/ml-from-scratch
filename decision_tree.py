import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
import pandas as pd

traning_data = [
    ['Green', 3, 'Apple'],
    ['Yellow', 3, 'Apple'],
    ['Red', 1, 'Grape'],
    ['Red', 1, 'Grape'],
    ['Yellow', 3, 'Lemon'],

]



header = ["color", "diameter", "label"]

def uniq_values(rows,col):

    return set([row[col] for row in rows])


uniq_values(traning_data, 0)


def class_count(rows):

    counts = {}
    for row in rows:
        label = row[-1]
        if label not in counts:
            counts[label] = 0
        
        counts[label] +=1

    return counts


class_count(traning_data)


def is_numeric(value):

    return isinstance(value, int) or isinstance(value, float)


is_numeric(7)