import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mat4py
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from PIL import Image as im


#data['labels'][i][0] corresponds to data['digits'][i]
directory = os.path.dirname(__file__)
data = mat4py.loadmat(directory + '/digits.mat')

labels = []
for item in data['labels']:
    labels.append(item[0])

labelset = pd.DataFrame(labels)
labelset.columns = ['labels']


keys = []
for i in range(0,400):
    keys.append('p' + str(i))

featureset = pd.DataFrame(data['digits'])


featureset.columns = keys


trainL, testL, trainD, testD = train_test_split(labelset, featureset, test_size = 0.5, random_state = 45)


model = GaussianNB()
model.fit(trainD,trainL.values.ravel())
print("Score before PCA: " + str(model.score(testD, testL.values.ravel())))


#Question 2:



lda = LDA()
newtrainD = lda.fit_transform(trainD,trainL)

count= 0
for mean in lda.means_: 
    thisavgArray = np.reshape(mean,(20,20))
    thisphotoData = im.fromarray(thisavgArray*255)
    thisphotoData = thisphotoData.convert("L")
    thisphotoData.save(directory + '/Report/LDAaverage'+str(count)+'.png')
    count += 1

rates=[]
nRange = [1,2,3,4,5,6,7,8,9]
count = 0
for n in nRange:
    print('Progress: ' + str(round(count/len(nRange)*100,2)) + '%')
    count += 1
    thisLDA = LDA(n_components = n)
    train = thisLDA.fit_transform(trainD,trainL)
    test = thisLDA.transform(testD)
    model = GaussianNB()
    model.fit(train,trainL.values.ravel())
    
    predictions = model.predict(test).tolist()
    success = 0
    for i in range(0,len(predictions)-1):
        if (predictions[i] == testL.values.tolist()[i][0]):
            success += 1
    successRate = (success/2500)*100
    rates.append(successRate)
    
plt.plot(nRange,rates)

plt.show()


    