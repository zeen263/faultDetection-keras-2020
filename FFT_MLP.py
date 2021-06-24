import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import keras
from keras.utils import np_utils
from keras.models import Sequential, load_model
from keras.layers.core import Dense, Dropout
from keras import regularizers
from sklearn.preprocessing import LabelEncoder


'''
sampling rate/2 = 측정할 주파수 범위
fft size = 저장되는 데이터의 갯수
magnitude = 해당 주파수 범위 신호의 세기

예를 들어 sampling rate가 1024, fft size가 256이라면
512Hz까지 측정 가능하며 저장되는 데이터의 갯수는 256개
하나의 데이터가 2Hz 범위의 정보를 가지게 됨

* 0~2Hz는 magnitude 값이 이상해서 뺐기 때문에
입력으로 사용할 데이터 크기는 255
'''

x_train=[]
y_train=[]

x_val=[]
y_val=[]

x_test=[]
y_test=[]

# 데이터셋 나누기...
def split(df):
    global x_train, y_train, x_val, y_val, x_test_double, y_test_double

    df.sample(frac=1).reset_index(drop=True)  # 데이터 셔플
    dataset = df.values
    x = dataset[:, 1:].astype(float)  # 행,열 순서 (모든 행, 1열부터 마지막 열까지)
    y = dataset[:, 0]


    e = LabelEncoder()
    e.fit(y)
    y = e.transform(y)
    y_encoded = np_utils.to_categorical(y)  # one-hot-encoding

    x = x.tolist()
    y = y.tolist()
    y_encoded = y_encoded.tolist()

    testset_cnt=[0,0,0,0,0,0]

    i = 0
    while testset_cnt[0] < 180:
        sample = int(y[i])
        if sample == 0:
            x_test.append(x[i])
            y_test.append(y_encoded[i])
            y.pop(i)
            y_encoded.pop(i)
            x.pop(i)

            testset_cnt[0] += 1
        else:
            i += 1
    print('check 0')
    i = 0
    while testset_cnt[1] < 180:
        sample = int(y[i])
        if sample == 1:
            x_test.append(x[i])
            y_test.append(y_encoded[i])
            y.pop(i)
            y_encoded.pop(i)
            x.pop(i)

            testset_cnt[1] += 1
        else:
            i += 1
    print('check 1')
    i = 0
    while testset_cnt[2] < 180:
        sample = int(y[i])
        if sample == 2:
            x_test.append(x[i])
            y_test.append(y_encoded[i])
            y.pop(i)
            y_encoded.pop(i)
            x.pop(i)

            testset_cnt[2] += 1
        else:
            i += 1
    print('check 2')
    i = 0
    while testset_cnt[3] < 180:
        sample = int(y[i])
        if sample == 3:
            x_test.append(x[i])
            y_test.append(y_encoded[i])
            y.pop(i)
            y_encoded.pop(i)
            x.pop(i)

            testset_cnt[3] += 1
        else:
            i += 1
    print('check 3')
    i = 0
    while testset_cnt[4] < 180:
        sample = int(y[i])
        if sample == 4:
            x_test.append(x[i])
            y_test.append(y_encoded[i])
            y.pop(i)
            y_encoded.pop(i)
            x.pop(i)

            testset_cnt[4] += 1
        else:
            i += 1
    print('check 4')
    i = 0
    while testset_cnt[5] < 180:
        sample = int(y[i])
        if sample == 5:
            x_test.append(x[i])
            y_test.append(y_encoded[i])
            y.pop(i)
            y_encoded.pop(i)
            x.pop(i)

            testset_cnt[5] += 1
        else:
            i += 1
    print('check 5')

    print('testset : ', len(x_test))

    data_len = len(x)
    for i in range(int(data_len * 0.3)):  # val : 테스트셋 빼고 남은거의 30%
        x_val.append(x.pop())
        y_val.append(y_encoded.pop())
    print('valset : ', len(x_val))

    data_len = len(x)
    for i in range(data_len):  # train
        x_train.append(x.pop())
        y_train.append(y_encoded.pop())
    print('trainset : ', len(x_train))

    print('total : ', len(x_test)+len(x_val)+len(x_train))




dataset_q = pd.read_csv('csv\\quiet.csv',header=None)
dataset_n = pd.read_csv('csv\\noise.csv',header=None)


dataset = pd.concat([dataset_n,dataset_q])
split(dataset)


x_val = np.array(x_val)
y_val = np.array(y_val)

x_train = np.array(x_train)
y_train = np.array(y_train)

x_test = np.array(x_test)
y_test = np.array(y_test)



model = Sequential()
model.add(Dense(128, activation='relu', input_dim=255))
model.add(Dropout(0.3))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(6, activation='softmax'))


model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

history = model.fit(x_train,y_train,
                    epochs=100, batch_size=50,
                    validation_data=(x_val, y_val))


loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss)+1)


# training loss
plt.rc('font',family='Times New Roman')
plt.plot(epochs, loss, label='Training loss : '+str(round(loss[-1],2)), color='#ff9124', linewidth=3)
plt.ylim(0,1)
plt.title('Training loss',fontsize=15 )
plt.legend(fontsize=15 )
plt.xlabel('epochs',fontsize=15 )
plt.ylabel('loss',fontsize=15 )

plt.show()


# val loss
plt.rc('font',family='Times New Roman')
plt.plot(epochs, val_loss, label='Validation loss : '+str(round(val_loss[-1],2)),color='#ff9124', linewidth=3)
plt.ylim(0,1)
plt.title('Validation loss',fontsize=15 )
plt.legend(fontsize=15 )
plt.xlabel('epochs',fontsize=15 )
plt.ylabel('loss',fontsize=15 )

plt.show()


acc = history.history['acc']
val_acc = history.history['val_acc']

#training acc
plt.rc('font',family='Times New Roman')
plt.plot(epochs, acc, label='Training acc : '+str(round(acc[-1],2)),color='#71b337', linewidth=3)
plt.ylim(0,1)
plt.title('Training accuracy',fontsize=15)
plt.legend(fontsize=15 )
plt.xlabel('epochs',fontsize=15 )
plt.ylabel('accuracy',fontsize=15 )
plt.show()

# val acc
plt.rc('font',family='Times New Roman')
plt.plot(epochs, val_acc, label='Validation acc : '+str(round(val_acc[-1],2)),color='#71b337',linewidth=3)
plt.ylim(0,1)
plt.title('Validation accuracy',fontsize=15)
plt.legend(fontsize=15 )
plt.xlabel('epochs',fontsize=15 )
plt.ylabel('accuracy',fontsize=15 )
plt.show()


predictions = model.predict(x_test).tolist()
right=[]
wrong=[]
result=[[],[],[],[],[],[]]

print()

for i in range(len(predictions)):
    pred = predictions[i]
    real = y_test[i].tolist()
    '''
    print('predict : [', end='')
    for j in pred:
        print('%.5f' %j, end=', ')
    print(']')

    print('it was  : [', end='')
    for j in real:
        print('%7d' %j, end=', ')
    print(']')
    '''
    index_was = real.index(max(real))

    if pred.index( max(pred) ) == index_was:
        #print('correct')
        right.append(index_was)
        result[index_was].append(1)
    else:
        #print('incorrect')
        wrong.append(index_was)
        result[index_was].append(0)

    #print()

print('{} test samples : correct {} / incorrect {}'.format(len(y_test),len(right),len(wrong)))
print('accuracy : %.3f %%' %((len(right)/len(y_test))*100) )


case_right=[]
case_wrong=[]

for i in range(len(result)):
    cnt_right = result[i].count(1)
    cnt_wrong = result[i].count(0)

    case_right.append(cnt_right)
    case_wrong.append(cnt_wrong)

print(case_right)
print(case_wrong)


label=['normal','stop','blade','dust','lubricant','wear']
index=np.arange(len(label))
wid=0.5
plt.rc('font',family='Times New Roman')
plt.bar(index, case_wrong,wid,color='#e36052')
plt.bar(index, case_right,wid,color='#66a62b', bottom=case_wrong)

plt.title('Prediction result',fontsize=12)
plt.xlabel('category',fontsize=12)
plt.ylabel('case',fontsize=12)
plt.legend(['Incorrect','Correct'],fontsize=12)
plt.xticks(index,label,fontsize=12)
plt.ylim(0,250)

dx=0.25
dy=5
plt.annotate(str(case_right[0]),xy=(0,185),fontsize=14,horizontalalignment='center')
plt.annotate(str(case_right[1]),xy=(1,185),fontsize=14,horizontalalignment='center')
plt.annotate(str(case_right[2]),xy=(2,185),fontsize=14,horizontalalignment='center')
plt.annotate(str(case_right[3]),xy=(3,185),fontsize=14,horizontalalignment='center')
plt.annotate(str(case_right[4]),xy=(4,185),fontsize=14,horizontalalignment='center')
plt.annotate(str(case_right[5]),xy=(5,185),fontsize=14,horizontalalignment='center')

plt.annotate(str(case_wrong[0]),xy=(0,case_wrong[0]+dy),fontsize=14,horizontalalignment='center')
plt.annotate(str(case_wrong[1]),xy=(1,case_wrong[1]+dy),fontsize=14,horizontalalignment='center')
plt.annotate(str(case_wrong[2]),xy=(2,case_wrong[2]+dy),fontsize=14,horizontalalignment='center')
plt.annotate(str(case_wrong[3]),xy=(3,case_wrong[3]+dy),fontsize=14,horizontalalignment='center')
plt.annotate(str(case_wrong[4]),xy=(4,case_wrong[4]+dy),fontsize=14,horizontalalignment='center')
plt.annotate(str(case_wrong[5]),xy=(5,case_wrong[5]+dy),fontsize=14,horizontalalignment='center')

plt.show()

