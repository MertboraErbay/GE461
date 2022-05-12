import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mat4py
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.decomposition import PCA
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


#Question 1:
mean_trainD = np.mean(trainD,axis = 0)
centered_trainD = trainD - mean_trainD
cov_matrix = np.cov(centered_trainD,rowvar = False)

eVal , eVec = np.linalg.eigh(cov_matrix)

sorted_eVal = eVal[np.argsort(eVal)[::-1]]

#plt.plot(sorted_eVal)
#plt.show()

pca = PCA(n_components = 400)
newtrainD = pca.fit_transform(trainD)

avgArray = np.reshape(pca.mean_,(20,20))
photoData = im.fromarray(avgArray*255)

photoData = photoData.convert("L")
photoData.save(directory + '/Report/averageDigit.png')

classes = {0:[],1:[],2:[],3:[],4:[],5:[],6:[],7:[],8:[],9:[]}

for i in trainD.index:
    classes[trainL[trainL.index==i].values[0][0]].append(trainD[trainD.index==i].values[0])

newpca = PCA(n_components = 60)
newpca.fit(trainD)
components = newpca.components_

for i in range(0,10):
    thisavgArray = np.reshape(components[i],(20,20))
    thisphotoData = im.fromarray(thisavgArray*255)
    thisphotoData = thisphotoData.convert("L")
    thisphotoData.save(directory + '/Report/average'+str(i)+'.png')

print("Photos Printed")

rates=[]
nRange = [40,45,50,55,60,65,70,75,80,85,90,95,100,105,110,115,120,125,130,135,140,145,150]
count = 0
for n in nRange:
    print('Progress: ' + str(round(count/len(nRange)*100,2)) + '%')
    count += 1
    thispca = PCA(n_components = n)
    trainPC = thispca.fit_transform(trainD)
    testPC = thispca.transform(testD)
    model = GaussianNB()
    model.fit(trainPC,trainL.values.ravel())
    
    predictions = model.predict(testPC).tolist()
    success = 0
    for i in range(0,len(predictions)-1):
        if (predictions[i] == testL.values.tolist()[i][0]):
            success += 1
    successRate = (success/2500)*100
    rates.append(successRate)
    
plt.plot(nRange,rates)

plt.show()
    