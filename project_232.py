from numpy import loadtxt
from keras.models import Sequential
from keras.layers import Dense

dataset=loadtxt('pokemon.csv', delimiter=',')

x=dataset[:,0:8]
y=dataset[:,8]

model=Sequential()
model.add(Dense(12,input_dim=8,activation='relu'))
model.add(Dense(8,activation='relu'))
model.add(Dense(1,activation='sigmoid'))

model.compile(loss='binary_crossentropy',metrics=['accuracy'])

model.fit(x,y,epochs=250,batch_size=100)

prediction= (model.predict(x)>0.5).astype("int32")
for i in range(785,800):
    print(f'{x[i].tolist()}=> {prediction[i]}expected{y[i]}')
