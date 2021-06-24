import sqlite3 as sq
import pandas as pd
import numpy as np
import random


def get_data(db, label=0):
    con = sq.connect(db + '.db')
    cursor = con.cursor()
    cursor.execute("SELECT * FROM Spectrogram")
    data = cursor.fetchall()  # type : list
    data = pd.DataFrame(data)[4]  # type : serise

    data = split_sum(data)  # 10개씩 묶어서 평균
    data = data / 10000  # 정규화

    datalist = []

    for i in data:
        datalist.append(np.insert(i, 0, str(label)))

    random.shuffle(datalist)

    return datalist


def split_sum(data):
    lst = []
    avg = []

    for i in data:  # split
        i = i[1:-1]
        line = i.split(', ')
        line.pop(0)  # 0Hz data is corrupted
        for j in range(len(line)):
            line[j] = float(line[j])
        lst.append(line)

    i = 0
    while i < len(lst):
        if len(lst) - i < 9: break
        arr = np.array([lst[i + j] for j in range(10)])
        avg.append(sum(arr))
        i += 10

    avg = np.array(avg) / 10

    return avg

# ================== 환경 소음 없는 데이터 ===================
normal = get_data('DB\\quiet\\normal', 0)
stop = get_data('DB\\quiet\\stop', 1)
fan = get_data('DB\\quiet\\fan', 2)
dust = get_data('DB\\quiet\\dust', 3)
bearing = get_data('DB\\quiet\\bearing', 4)
spindle = get_data('DB\\quiet\\spindle', 5)

totaldata = normal + fan + dust + bearing + stop + spindle
random.shuffle(totaldata)
dataframe = pd.DataFrame(totaldata)
dataframe.to_csv('quiet.csv', header=False, index=False)


# ================== 환경 소음 있는 데이터 ===================
normal = get_data('DB\\noisy\\normal', 0)
stop = get_data('DB\\noisy\\stop', 1)
fan = get_data('DB\\noisy\\fan', 2)
dust = get_data('DB\\noisy\\dust', 3)
bearing = get_data('DB\\noisy\\bearing', 4)
spindle = get_data('DB\\noisy\\spindle', 5)

totaldata = normal + fan + dust + bearing + stop + spindle
random.shuffle(totaldata)
dataframe = pd.DataFrame(totaldata)
dataframe.to_csv('noisy.csv', header=False, index=False)