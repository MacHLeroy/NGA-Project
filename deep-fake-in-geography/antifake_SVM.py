#-*-coding:utf-8-*-

import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.model_selection import cross_val_score

#absolute path & before excuting the script, I deleted all the unnecessary columns except "isFake" and 26 features
inputFile='E:/desktop/antiFake_result.csv'
df=pd.read_csv(inputFile)
data=df.values
data[:,1:26]=(data[:,1:26]-np.min(data[:,1:26],axis=0))/(np.max(data[:,1:26],axis=0)-np.min(data[:,1:26],axis=0))
np.random.shuffle(data)

#all
# spatial=data[:,15:22]
# histogram=data[:,1:14]
# frequency=data[:,23:26]

#strict salient
spatial=data[:,(15,19,22)]
histogram=data[:,(1,3,6,9,11,12)]
frequency=data[:,(23,24,26)]

#normal salient
# spatial=data[:,15:22]
# histogram=data[:,1:14]
# frequency=data[:,(23,24,26)]

clf = svm.SVC(kernel='linear', C=1)
calculist=[]
calculist.append(spatial)
calculist.append(histogram)
calculist.append(frequency)
calculist.append(np.hstack((spatial,histogram)))
calculist.append(np.hstack((spatial,frequency)))
calculist.append(np.hstack((histogram,frequency)))
calculist.append(np.hstack((spatial,histogram,frequency)))
for calcu in calculist:
    scores_F1 = cross_val_score(clf, calcu, data[:,0], cv=5, scoring='f1')
    scores_Accuracy=cross_val_score(clf, calcu, data[:,0], cv=5, scoring='accuracy')
    scores_Precision=cross_val_score(clf, calcu, data[:,0], cv=5, scoring='precision')
    scores_Recall=cross_val_score(clf, calcu, data[:,0], cv=5, scoring='recall')
    print([scores_F1.mean(),scores_Accuracy.mean(),scores_Precision.mean(),scores_Recall.mean()])


