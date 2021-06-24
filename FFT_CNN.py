import numpy as np, matplotlib.pyplot as plt, pandas as pd

from random import shuffle
from keras import layers
from keras import models
from keras.utils import np_utils, to_categorical
from keras.layers.core import Dense, Dropout
from PIL import Image, ImageOps
from sklearn.preprocessing import LabelEncoder

train = []
val = []
test = []
test_double = []

# 데이터 수는 변동 가능
datacnt = {'normal':1516,'stop':2040,'fan':1174,'dust':1808,'bearing':1502,'wear':1737}

# 레이블 : 0 normal / 1 stop / 2 fan / 3 dust / 4 bearing / 5 wear
#            1516     2040      1174     1808     1502      1737

# 상황별로 테스트셋 180개를 먼저 분리, 남는것의 70%를 학습셋 30%를 검증셋으로 사용


def png2np(path):
    for i in range(1,181):  # 테스트셋에 집어넣기
        if i < 10:
            idx = '00' + str(i)
        elif i < 100:
            idx = '0' + str(i)
        else:
            idx = str(i)

        img = Image.open('capture/' + path + '/img' + idx + '.png').convert('L')
        data = np.array(img)


        data = data / 255

        lbl = None
        if path == 'normal':
            lbl = 0
        elif path == 'stop':
            lbl = 1
        elif path == 'fan':
            lbl = 2
        elif path == 'dust':
            lbl = 3
        elif path == 'bearing':
            lbl = 4
        elif path == 'wear':
            lbl = 5

        test.append((lbl, data))

    train_len = int(datacnt[path]*0.7)
    for i in range(181,181+train_len): # 학습셋에 집어넣기
        if i<10: idx = '00'+str(i)
        elif i<100: idx = '0'+str(i)
        else: idx = str(i)

        img = Image.open('capture/'+ path + '/img'+ idx +'.png').convert('L')
        data = np.array(img)

        #plt.imshow(data)
        #plt.show()

        data = data/255 # 주의: 이미지의 차원은 세로x가로x채널 형태로 표시됨(80, 300, 3)

        lbl = None
        if path == 'normal': lbl = 0
        elif path == 'stop': lbl = 1
        elif path == 'fan': lbl = 2
        elif path == 'dust': lbl = 3
        elif path == 'bearing': lbl = 4
        elif path == 'wear': lbl = 5

        train.append((lbl, data))

    for i in range(181+train_len, datacnt[path]):  # 검증셋에 집어넣기
        if i < 10:
            idx = '00' + str(i)
        elif i < 100:
            idx = '0' + str(i)
        else:
            idx = str(i)

        img = Image.open('capture/' + path + '/img' + idx + '.png').convert('L')
        data = np.array(img)


        data = data / 255

        lbl = None
        if path == 'normal':
            lbl = 0
        elif path == 'stop':
            lbl = 1
        elif path == 'fan':
            lbl = 2
        elif path == 'dust':
            lbl = 3
        elif path == 'bearing':
            lbl = 4
        elif path == 'wear':
            lbl = 5

        val.append((lbl, data))



png2np('normal')
png2np('stop')
png2np('fan')
png2np('dust')
png2np('bearing')
png2np('wear')

shuffle(train); shuffle(val); shuffle(test);


train_lbl = []; train_data = []
val_lbl = []; val_data = []
test_lbl = []; test_data = []

for item in train:
    train_lbl.append(item[0])
    train_data.append(item[1].reshape(80,300,1))

for item in val:
    val_lbl.append(item[0])
    val_data.append(item[1].reshape(80,300,1))
    
for item in test:
    test_lbl.append(item[0])
    test_data.append(item[1].reshape(80,300,1))


e = LabelEncoder()
e.fit(train_lbl)
train_lbl = e.transform(train_lbl)
train_lbl = np_utils.to_categorical(train_lbl)  # one-hot-encoding

e.fit(val_lbl)
val_lbl = e.transform(val_lbl)
val_lbl = np_utils.to_categorical(val_lbl)

e.fit(test_lbl)
test_lbl = e.transform(test_lbl)
test_lbl = np_utils.to_categorical(test_lbl)


train_data = np.array(train_data); train_lbl = np.array(train_lbl)
val_data = np.array(val_data); val_lbl = np.array(val_lbl)
test_data = np.array(test_data); test_lbl = np.array(test_lbl)


print(np.shape(train_data[0]))

model = models.Sequential()
model.add(layers.Conv2D(6,(5,5),activation='relu',input_shape=(80,300,1))) # 이미지는 세로, 가로, 채널 순
model.add(layers.MaxPooling2D((2,2))) # (2,2)는 풀링 필터의 크기, 스트라이드 따로 안 정하면 풀링 필터 크기와 같다 (풀링하면 이미지 가로세로가 반으로 줄어든다)
model.add(layers.Conv2D(12,(5,5),activation='relu')) # (5,5)는 필터의 크기
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(24,(5,5),activation='relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Flatten())
model.add(layers.Dropout(0.5))
model.add(layers.Dense(32, activation='relu'))
model.add(layers.Dense(6, activation='softmax')) # 6개 카테고리 구별


model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

history = model.fit(train_data,train_lbl,
                    epochs=50, batch_size=50,
                    validation_data=(val_data, val_lbl))

print(model.summary())

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


predictions = model.predict(test_data).tolist()
right=[]
wrong=[]
result=[[],[],[],[],[],[]]


print()
for i in range(len(predictions)):
    pred = predictions[i]
    real = test_lbl[i].tolist()
    '''
    print('predict : [', end='')
    for j in pred:
        print('%.5f' %j, end=', ')
    print(']', end = '    ')

    print('it was : [', end='')
    for j in real:
        print('%2d' %j, end=', ')
    print(']', end = '    ')
    '''

    index_was = real.index(max(real))

    if pred.index( max(pred) ) == real.index( max(real) ):
        #print('(correct)')
        right.append( real.index( max(real) ))
        result[index_was].append(1)
    else:
        #print('(incorrect)')
        wrong.append( real.index( max(real) ))
        result[index_was].append(0)

    #print()

print('{} test samples : correct {} / incorrect {}'.format(len(test_lbl),len(right),len(wrong)))
print('accuracy : %.3f %%' %((len(right)/len(test_lbl))*100) )

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