# Импортируем библиотеки
import pandas as pd
import numpy as np

# графические библиотеки
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# библиотеки машинного обучения
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# отображать по умолчанию длину Датафрейма
pd.set_option("display.max_rows", 9, "display.max_columns", 9)

# библиотека взаимодействия с интерпретатором
import sys
if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")

import os

# загрузка файлов.
#PATH = 'E:\Kaggle\Motorica_2' 
PATH = ""

#загрузка обучающей выборки и меток классов
X_train = np.load(os.path.join(PATH, 'X_train.npy'))
y_train = pd.read_csv(os.path.join(PATH, 'y_train.csv'), sep='[-,]',  engine='python') 
y_train_vectors = y_train.pivot_table(index='sample', columns='timestep', values='class')

#загрузка тестовой выборки
X_test = np.load(os.path.join(PATH, 'X_test.npy'))

def get_y_train():
    y_train = pd.read_csv(os.path.join(PATH, 'y_train.csv'), sep='[-,]',  engine='python')
    y_train_vectors = y_train.pivot_table(index='sample', columns='timestep', values='class').values
    return y_train_vectors

def get_x_train():
    X_train = np.load(os.path.join(PATH, 'X_train.npy'))
    return get_x_train