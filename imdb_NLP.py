import os
import numpy as np
import matplotlib.pyplot as plt
from keras import models,layers
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequencs

#Reading the data from a folder on the local machine
imdb_dir = '/Users/sumanth/Documents/PythonLearning/MachineLearning/imdb_dataSet'
train_dir = os.path.join(imdb_dir,'train')
labels = []; texts = []
for label_type in ['neg','pos']:
    dir_name = os.path.join(train_dir,label_type)
    for fname in os.listdir(dir_name):
        if fname[-4:] == '.txt':
            f = open(os.path.join(dir_name,fname))
            texts.append(f.read())
            f.close()
            if label_type == 'neg':
                labels.append(0)
            else:
                labels.append(1)

#Tokenizing the data
maxlen = 100 #Cut-off after 100 words
training_samples = 200
validation_samples = 10000
max_words = 10000
tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
word_index = tokenizer.word_index
print("Found %s unique tokens." %len(word_index))

data = pad_sequences(sequences,maxlen=maxlen)

labels = np.asarray(labels)
print("Shape of data tensor:",data.shape)
print("Shape of label tensor:",labels.shape)

indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]
x_train = data[:training_samples]
y_train = labels[:training_samples]
x_val = data[training_samples:training_samples+validation_samples]
y_val = labels[training_samples:training_samples+validation_samples]

#Prepare the models
model = models.Sequential()
model.add(layers.Embedding(max_words,8,input_length=maxlen))
model.add(layers.Flatten())
model.add(layers.Dense(1,activation='sigmoid'))
model.compile(optimizer='rmsprop',loss='binary_crossentropy',metrics=['acc'])
model.summary()

#Fit the model
history = model.fit(x_train,y_train,epochs=10,batch_size=32,validation_data=(x_val,y_val))

#Plot the results
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1,len(acc)+1)
plt.plot(epochs,acc,'bo',label='Training Accuracy')
plt.plot(epochs,val_acc,'b',label='Validation Accuracy')
plt.title('Training and validation accuracy')
plt.legend()
plt.grid()
plt.show()
