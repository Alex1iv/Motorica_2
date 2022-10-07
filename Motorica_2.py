

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

import __init__
__init__.X_train

gestures = ['open',  # 0
            'мизинец',  # 1
            'безымянный палец',  # 2
            'средний палец',  # 3
            'пистолет',  # 4
            'указательный палец',  # 5
            'большой палец',  # 6
            'ok',  # 7
            'grab',  # 8
]


def stratified_cross_valid(model: LogisticRegression, 
                           X_train: pd.DataFrame, y_train: pd.Series, 
                           n: int, metric: str) -> None:
    """
    Функция оценки модели на кросс-валидации: вывод графика с результатами 
    кросс-валидации, кривой достаточности данных и таблицы с результатами
    """
    print(model)
    cv_splitter = StratifiedKFold(n_splits = n)
    
    metric_table = pd.DataFrame()

    i = 0
                    
    cv_res = cross_validate(
        model, 
        X_train, 
        y_train, 
        scoring = metric, 
        n_jobs = -1, 
        cv = cv_splitter, 
        return_train_score = True, 
        verbose = 0)

    cv_train = cv_res['train_score'].mean()
    cv_test = cv_res['test_score'].mean()


    metric_table.loc[i, 'cv_train'] = cv_train
    metric_table.loc[i, 'cv_test'] = cv_test

                    
    metric_table['cv_dif'] = metric_table['cv_train'] - metric_table['cv_test']

    #результаты кросс-валидации
    fig = sns.pointplot(x=np.arange(n), y=cv_res['train_score'], color = 'r')
    fig = sns.pointplot(x=np.arange(n), y=cv_res['test_score'], color = 'g')
    fig.set_title('Результаты кросс-валидации', fontsize=16)
    fig.set_xlabel('Порядковый номер части совокупности')
    fig.set_ylabel('Показатель качества модели\n f-score')
    #plot.set_ylim(0.4, 1.1)
    fig.grid() # показать сетку
    plt.show()

    # кривая обучения
    result = []

    s = len(X_train)
    p = len(X_train) // (n + 1)
    for i in np.arange(p, s - p + 1, p):
        model.fit(X_train.iloc[:i], y_train.iloc[:i].values.ravel())  # + .values.ravel()
        predict = model.predict(X_train.iloc[i:i+p])
        res = f1_score(y_train.iloc[i:i+p], predict, average='macro')
        result.append(res)

    fig = sns.pointplot(x=np.arange(p, s - p + 1, p), y=result)
    fig.set_title('Кривая обучения', fontsize=16)
    fig.set_xlabel('Количество объектов для обучения')
    fig.set_ylabel('Показатель качества  модели\n f-score')
    fig.set_ylim(0.4, 1.05)
    fig.grid() # показать сетку
    plt.show()
    print ()
    
    print(metric_table.sort_values(by ='cv_test'))
    print ()


def read_gests(xtrain: np.array, ytrain: pd.DataFrame, 
               t_begin: int, t_end: int) -> pd.DataFrame: # (X_train, y_train, 50, 80)
    '''Конвертируем ndarray в DataFrame, где к колонкам вида 'sensor_{s}' добавляем
    'Class' - реализованный в моменте жест в соответствии с y_train,
    'b_e' - категории временных интервалов: 'begin', 'pass', 'end', -
            определенных по t_begin и t_end, чтобы выделить время перехода,
    'Class_name' - наименование жеста
    '''
    samples = len(xtrain)
    read_xtrain_df = pd.DataFrame(columns=[f'sensor_{s}' for s in range(40)] 
                                  + ['Class', 'b_e']) 
    for samp in range(samples):
        read_xtrain_df_s = pd.DataFrame(data=xtrain[samp], 
                                        index=[f'sensor_{s}' for s in range(40)],
                                        columns=[t for t in range(100)]).T
        y = pd.DataFrame(ytrain[ytrain['sample'] == samp]['class'])
        y.index = [t for t in range(100)]
        read_xtrain_df_s['Class'] = y['class']
        read_xtrain_df_s['b_e'] = read_xtrain_df_s.index
        read_xtrain_df_s['b_e'] = read_xtrain_df_s['b_e'].apply(
            lambda t: 'begin' if t in range(t_begin) 
            else ('pass' if t in range(t_begin, t_end) else 'end'))
        read_xtrain_df = pd.concat([read_xtrain_df, read_xtrain_df_s])
    read_xtrain_df['Class_name'] = read_xtrain_df['Class'].apply(
            lambda cl: gestures[cl])
    return read_xtrain_df


def box_sens(sens: int, read_xtrain_df: pd.DataFrame) -> None: 
    # (5, df[df['b_e'] != 'pass'].loc[:, ['sensor_5', 'b_e', 'Class_name']])
    '''строит боксплот для датчика sens в разрезе жестов, 
    агрегируя по всем наблюдениям'''
    plt.style.use('ggplot')
    fig = plt.figure(figsize=(16, 10))
    boxplot = sns.boxplot(data=read_xtrain_df, 
                          x=f'sensor_{sens}', y='Class_name', hue='b_e', 
                          orient='h', width=0.9, dodge=True
                          )

    
def box_sens_all(read_xtrain_df: pd.DataFrame) -> None:  # (df[df['b_e'] != 'pass'])
    '''строит боксплоты для всех 40 датчиков в разрезе жестов, 
    агрегируя по всем наблюдениям'''
    plt.style.use('ggplot')
    fig, axes = plt.subplots(40, 1, figsize=(16, 400))
    for sens in range(40):
        boxplot = sns.boxplot(data=read_xtrain_df, x=f'sensor_{sens}', y='Class_name', 
                              ax=axes[sens], hue='b_e', orient='h', 
                              width=0.9, dodge=True, showmeans=True)


def x_dataframe(x_base: np.array) -> pd.DataFrame:   # (X_train)
    '''Переводит np.array X_train в формат DataFrame.
    Столбцы - датчики, индексы в формате "наблюдение-время"'''
    x_df = pd.DataFrame(columns=[f'sensor_{s}' for s in range(40)])
    for samp in range(len(x_base)):
        x_samp = pd.DataFrame(data=x_base[samp], 
                              index=[f'sensor_{s}' for s in range(40)], 
                              columns=[f'{samp}-{t}' for t in range(100)]).T
        x_df = pd.concat([x_df, x_samp], ignore_index=False)
    return x_df


def data_out_transit(xdf: pd.DataFrame, ydf: pd.DataFrame,
                     t_transit: list) -> pd.DataFrame:  
                                    # (x_df, y_train_raw, t_transit)
    '''Исключает из преобразованных в DataFrame данных X_train и y_train
    строки для временных интервалов, заданных в t_transit.
    Используется для обрезки данных за время перехода (движения).
    Возвращает две DataFrame'''
    xdf_ot = pd.DataFrame(columns=[f'sensor_{s}' for s in range(40)])
    ydf_ot = pd.DataFrame(columns=['Class', 'sample', 'timestep'])
    for row in range(ydf.shape[0]):
        t = int(ydf['timestep'][row])
        if t not in t_transit:
           xdf_ot = pd.concat([xdf_ot, xdf[xdf.index == ydf['sample-timestep'][row]]], 
                              ignore_index=False)
           ydf_ot = pd.concat([ydf_ot, ydf[ydf.index == row]], ignore_index=False)
    return xdf_ot, ydf_ot        

def zero_levels_base(xtrain: np.array)-> pd.DataFrame:    # (X_train)
    '''Формирует из X_train данные по жестам "0-0" 
    для получения статистики по датчикам'''
    samples = len(xtrain)
    #y_train_all_00 = [samp + i for samp in range(0, samples, 18) for i in (0, 1)]
    y_train_2nd_00 = [samp for samp in range(1, samples, 18)]
    read_xtrain_df = pd.DataFrame(columns=[f'sensor_{s}' for s in range(40)]) 
    for samp in y_train_2nd_00:
        read_xtrain_df_s = pd.DataFrame(data=xtrain[samp], 
                                    index=[f'sensor_{s}' for s in range(40)],
                                    columns=[t for t in range(100)]).T
        read_xtrain_df = pd.concat([read_xtrain_df, read_xtrain_df_s])
    return read_xtrain_df        
            

def zero_levels_stat(xtrain: np.array, qtile_l: int, qtile_h: int) -> pd.DataFrame:     
    # (X_train, quantile_low, quantile_high) 
    '''Строит таблицу со статистическими данными для датчиков,
    полученными на жестах "0-0"'''   
    stat_0 = pd.DataFrame(data=[zero_levels_base(xtrain).median(),
                                zero_levels_base(xtrain).quantile(qtile_l/100),
                                zero_levels_base(xtrain).quantile(qtile_h/100)],
                          index=['median', 'low', 'high'])
    return stat_0


def sensor_intervals(s_ql: int, s_qh: int, intv: int) -> list: 
    #(levels_st.iloc[1, s], levels_st.iloc[2, s], intervals)
    '''Создает список интервалов на основании выбранных квантилей и
    количества интервалов'''
    interval_list = [0] + [s_ql + (s_qh - s_ql) * (i - 3) for i in range(intv - 1)] \
        + [s_qh * 10]
    return interval_list


def sensors_mask(levels_st: pd.DataFrame, intervals: int) -> list: 
    # (zero_levels_st, intervals)
    '''Из списка интервалов и статистических данных для каждого датчика
    создает маску для преобразования данных в категориальную форму'''
    mask_list = [sensor_intervals(levels_st.iloc[1, s], 
                                  levels_st.iloc[2, s], intervals) for s in range(40)]
    return mask_list


def compare_to_intervals(x: float, s_mask: list, intervals: int) -> float:   
                                    # (x, ss_mask, intervals)
    '''Функция для преобразования в категориальную форму'''
    x = [i for i in range(intervals) if s_mask[i] <= x and x < s_mask[i+1]]
    return x[0]


def intervals_freq(x_base: pd.DataFrame, intervals: int) -> pd.DataFrame:  
                                    # (X_train_preproc, intervals)
    '''Показывает количество наблюдений датчиков в каждом интервале'''
    xif = pd.DataFrame(data=[[x_base[x_base[f'sensor_{s}'] == i][f'sensor_{s}'].count() 
                             for s in range(40)] for i in range(intervals)],
                       columns=[f'sensor_{s}' for s in range(40)],
                       index=[i for i in range(intervals)])
    return xif







def privet(name):   # print 'privet' to a given name
    print(f'privet {name}')

def get_y_train():
    y_train = pd.read_csv(os.path.join(PATH, 'y_train.csv'), sep='[-,]',  engine='python')
    y_train_vectors = y_train.pivot_table(index='sample', columns='timestep', values='class').values
    return y_train_vectors

def get_x_train():
    np.load(os.path.join(PATH, 'X_train.npy'))
    return get_x_train

def get_test_id(id):
    """
    #Функция отображения списка наблюдений.
    #Аргументы функции: список из номера теста (timestep) и класса жеста
    """
    y_train_vectors = get_y_train()

    samples = list()
    for i in range(y_train_vectors.shape[0]):
        if y_train_vectors[i][0]==int(id[0]) and y_train_vectors[i][-1]==int(id[2]):
            samples.append(i)

    print(f"Наблюдения жеста {id}: {str(samples)}")
    samples = pd.Series(data=samples, name=f'{str(id)}', index=[id]*len(samples))
    return samples



def get_sensor_list(id, print_active=False, print_reliable=False):
    """
    Функция печати и импорта в память всех номеров датчиков
    Аргумент функции - номер наблюдения. 
    """
    import __init__
    X_train = __init__.X_train

    df = pd.DataFrame(data = X_train[id], index = [s for s in range(X_train.shape[1])], 
                        columns = [s for s in range(X_train.shape[2])]
    )
    
    # Создадим список индексов активных и пассивных датчиков. Среднее значение сигнала не превышает 200 единиц.
    active_sensors, passive_sensors, reliable_sensors, unreliable_sensors  = list(), list(), list(), list()
    
    for i in range(X_train.shape[1]):
        # если средняя амплитуда превышает 200, то добавляем индекс в 'active_sensors'
        if df.iloc[i].mean() > 200:
            active_sensors.append(i)
                   
            # Если разница между абсолютными средними значениями за последние 15 сек и первые 60 сек превышает 200,
            # то датчики заносим в список надежных. Остальные датчики с малой амплитудой - в список ненадёжных. 
            if abs(df.iloc[i][0:49].mean() - df.iloc[i][85:].mean()) > 200:
                reliable_sensors.append(i)
            else:
                unreliable_sensors.append(i)
        else:
            passive_sensors.append(i)
  
      
    if print_active is True:
        print(f"Активные датчики наблюдения " + str(id) +": ", active_sensors)
        print(f"Пассивные датчики наблюдения " + str(id) +":", str(passive_sensors))
    elif print_reliable is True:
        print(f"Датчики с большой амплитудой, наблюдения " + str(id) +": ", reliable_sensors)
        print(f"Датчики с малой амплитудой, " + str(id) +": ", unreliable_sensors)  
    
    return active_sensors, passive_sensors, reliable_sensors, unreliable_sensors



def get_all_sensors_plot(id, plot_counter):
    """
    Функция построения диаграммы показания датчиков. Аргумент функции - номер наблюдения и порядковый номер рисунка
    """
    import __init__
    X_train = __init__.X_train

    fig = px.line(data_frame=X_train[id].T)
    
    fig.update_layout(
        title=dict(text=f'Рис. {plot_counter}'+' - наблюдение ' + str(id), x=.5, y=0.05, xanchor='center'), 
        xaxis_title_text = 'Время, сек', yaxis_title_text = 'Показатель', # yaxis_range = [0, 3000],
        legend_title_text='Индекс <br>датчика',
        width=600, height=400,
        margin=dict(l=100, r=60, t=80, b=100),
    )

    fig.show()

def get_gest_plot(id, plot_counter):
    """
    Функция построения диаграммы классификации жеста. Аргумент функции - номер наблюдения и порядковый номер рисунка
    """
    import __init__
    y_train = __init__.y_train
    
    y_k = y_train[id*100:(id+1)*100].reset_index().T
    # смотрим скользящее STD

    time_stp = 0
    time_end = 100
    
    # Изображаем y_k
    y_k_stp = y_k[range(time_stp, time_end)]
    fig = plt.figure(figsize=(6, 2))
    ax = fig.add_axes([0.5, 0, 1, 1]) # ax = fig.add_axes([left, bottom, width, height])
    ax.plot(list(y_k_stp.columns), y_k_stp.loc['class'].values)
    
    ax.set_title(f'Рис. {plot_counter}'+' - Изменение класса жеста', y=-0.5, fontsize=16)
    ax.set_xlabel('Время')
    ax.set_ylabel('Класс')
    
    plt.show()

def get_active_passive_sensors_plot(id, plot_counter):
    """
    Функция построения графика показаний активных и пассивных датчиков.
    Аргумент функции - номер наблюдения и порядковый номер рисунка. 
    """
    
    import __init__
    X_train = __init__.X_train
    active_sensors, passive_sensors, reliable_sensors, unreliable_sensors = get_sensor_list(id) # списки сенсоров не печатаем

    
    df = pd.DataFrame(data = X_train[id], 
        index = [s for s in range(X_train.shape[1])], 
        columns = [s for s in range(X_train.shape[2])]
    )
    
    #get_sensor_list(id, False)

    df_3 = pd.DataFrame(X_train[id][active_sensors].T, columns=active_sensors)
    df_4 = pd.DataFrame(X_train[id][passive_sensors].T, columns=passive_sensors)

    fig = make_subplots(rows=1, cols=2, 
                        subplot_titles=('активные датчики', 'пассивные датчики')
    )
    
    for i in df_3.columns: fig.add_trace(go.Scatter(x=df_3.index, y=df_3[i], name=str(df[i].name)), row=1, col=1)

    for i in df_4.columns: fig.add_trace(go.Scatter(x=df_4.index, y=df_4[i], name=str(df[i].name)), row=1, col=2)

    fig.update_layout(title={'text':f'Рис. {plot_counter}'+' - Активные и пассивные датчики наблюдения ' + str(id), 'x':0.5, 'y':0.05}
    )

    fig.update_layout(width=1000, height=400, legend_title_text ='Номер датчика',
                        xaxis_title_text  = 'Время',  yaxis_title_text = 'Сигнал датчика', yaxis_range=  [0, 3000], 
                        xaxis2_title_text = 'Время', yaxis2_title_text = 'Сигнал датчика', yaxis2_range= [0 , 200],
                        margin=dict(l=100, r=60, t=80, b=100), 
                        #showlegend=False # легенда загромождает картинку
    )

    fig.show()

def get_amplitude(id, plot_counter):
    """
    Функция отображения гистограммы амплитуд сильных и слабых датчиков
    Аргумент функции - номер наблюдения и порядковый номер рисунка
    """
    import __init__
    X_train = __init__.X_train
        
    df = pd.DataFrame(data = X_train[id], index = [s for s in range(X_train.shape[1])], 
                        columns = [s for s in range(X_train.shape[2])]
    )
    
    df_mean = list() # cписок средних абсолютных амплитуд датчиков

    for i in range(X_train.shape[1]):
        # средняя величина сигнала
        value = df.iloc[i].mean()
        # разница между абсолютными средними значениями за последние 15 сек и первые 50 сек 
        amplitude = abs(df.iloc[i][85:].mean() - df.iloc[i][0:49].mean())

        #если средняя величина сигнала превышает 200 
        if value > 200 :
            df_mean.append(amplitude)

    
    df_mean = pd.DataFrame({f'amplitude_{id}':df_mean}, index=range(len(df_mean)))

    #print(df_mean.sort_values(df_mean.columns[0], ascending = False).head(2), df_mean.sort_values(df_mean.columns[0], ascending = False).tail(2))
    
    
    fig = px.histogram(
        data_frame=df_mean, 
        nbins=7,
        opacity=0.5, # непрозрачность 
        marginal='box'
    )

        
    fig.update_layout(
        title=dict(text=f'Рис. {plot_counter} - Распределение амплитуд датчиков по группам', x=.5, y=0.01),
        width=500, height=300,
        xaxis_title_text = 'Амплитуда',
        yaxis_title_text = 'Количество датчиков <br>в группе',
        template="simple_white",
        showlegend=False # легенда загромождает картинку
    )
    fig.show();

def get_strong_weak_sensors_plot(id, plot_counter):
    """
    Функция отображения диаграммы амплитуд сильных и слабых датчиков
    Аргумент функции - номер наблюдения
    """
    import __init__
    X_train = __init__.X_train
    active_sensors, passive_sensors, reliable_sensors, unreliable_sensors = get_sensor_list(id) # списки сенсоров не печатаем
    
    # вызов функции загрузки списка сенсоров  
    get_sensor_list(id)
    X_train=np.load(os.path.join(PATH, 'X_train.npy'))
    
    df = pd.DataFrame(data = X_train[id], 
        index = [s for s in range(X_train.shape[1])], 
        columns = [s for s in range(X_train.shape[2])]
    )

    df_ = {}
    df_1 = pd.DataFrame(X_train[id][reliable_sensors].T, columns=reliable_sensors)
    df_2 = pd.DataFrame(X_train[id][unreliable_sensors].T, columns=unreliable_sensors)

    

    fig = make_subplots(rows=1, cols=2, subplot_titles=('сильные сигналы', 'слабые сигналы'))
    
    for m in df_1.columns:
        fig.add_trace(go.Scatter(x=df_1.index, y=df_1[m], name=str(df_1.iloc[m].name)),
        row=1, col=1 , 
    )

    for k in df_2.columns:
        fig.add_trace(go.Scatter(x=df_2.index, y=df_2[k], name=str(df_1.iloc[k].name)), 
        row=1, col=2
    )
    
    fig.update_layout(title={'text':f'Рис. {plot_counter}'+' - Датчики с сильными и слабыми сигналами, наблюдение ' + str(id), 'x':0.5, 'y':0.01}
    )

    fig.update_layout(width=1000, height=400, legend_title_text ='Номер датчика',
                        xaxis_title_text  = 'Время', yaxis_title_text  = 'Сигнал датчика', yaxis_range = [0, 3500],
                        xaxis2_title_text = 'Время', yaxis2_title_text = 'Сигнал датчика', yaxis2_range= [0, 3500],
                        #showlegend=False # легенда загромождает картинку
    )

    fig.show()
    print(' ')
    print(f"Датчики с большой амплитудой, наблюдение " + str(id) +": ", reliable_sensors)
    print(f"Датчики с малой амплитудой, наблюдение " + str(id) +": ", unreliable_sensors) 
    


def get_sensors_in_all_tests_plot(arg1, arg2, plot_counter):
    """
    Функция вывода диаграммы показания отдельных датчиков во всех наблюдениях конкретного жеста
    Аргументом функции является строка - список датчиков
    """
    #функция отбора наблюдений в переменную 'samples' 
    
    samples = get_test_id(arg1)
    sensor_list = arg2 
    X_train=np.load(os.path.join(PATH, 'X_train.npy'))

    df_selected = pd.DataFrame(columns=range(100))
    for sample in samples:
        for sensor in sensor_list:
            #df = pd.Series([X_train[test,sensor]], index=[str(f"{sensor}")], name=str(f"{test}"))
            df = pd.DataFrame(X_train[sample,sensor]).T #, columns=range(100)
            
            df_selected = pd.merge(df_selected, df, how='outer') # , how='outer'
    # определим сколько графиков выводить
    len(arg2)
    if len(arg2)%2==0:

        fig = make_subplots(rows=2, cols=2, subplot_titles=(
            f"датчик {arg2[0]}", f"датчик {arg2[1]}", 
            f"датчик {arg2[2]} ", f"датчик {arg2[3]}")
        )

        df_1 = df_selected.iloc[0::len(sensor_list)].T
        df_2 = df_selected.iloc[1::len(sensor_list)].T
        df_3 = df_selected.iloc[2::len(sensor_list)].T
        df_4 = df_selected.iloc[3::len(sensor_list)].T

        for i in df_1.columns: 
            fig.add_trace(go.Scatter(x=df_1.index, y=df_1[i]), row=1, col=1)

        for i in df_2.columns:
            fig.add_trace(go.Scatter(x=df_2.index, y=df_2[i]), row=1, col=2)

        for i in df_3.columns:
            fig.add_trace(go.Scatter(x=df_3.index, y=df_3[i]), row=2, col=1)
            
        for i in df_4.columns:
            fig.add_trace(go.Scatter(x=df_4.index, y=df_4[i]), row=2, col=2)
            


        fig.update_layout(height=800, width=1000, #yaxis_type='log', 
                        title_text="Показания датчиков " + str(sensor_list) + " в наблюдениях жеста 0-1", title_xanchor='left', title_font=dict(size = 22),
                        xaxis_title_text  = 'Время', yaxis_title_text  = 'Сигнал датчика', yaxis_range =[0 ,3500], 
                        xaxis2_title_text = 'Время', yaxis2_title_text = 'Сигнал датчика', yaxis2_range=[0 ,3500], 
                        xaxis3_title_text = 'Время', yaxis3_title_text = 'Сигнал датчика', yaxis3_range=[0 ,3000], 
                        xaxis4_title_text = 'Время', yaxis4_title_text = 'Сигнал датчика', yaxis4_range=[0 ,3000], 
                        showlegend=False # легенда загромождает картинку
        )
        fig.update_layout(title=dict(text=f'Рис. {plot_counter}'+' - Сигнал датчиков во всех наблюдениях жеста ' + str(arg1), x=0.5, y=0.01, xanchor='center')
        )
        fig.show()
  


