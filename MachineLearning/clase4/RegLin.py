'''
Created on Sep 1, 2016

@author: Javier Quiroz
'''
from sklearn.linear_model.base import LinearRegression
import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn import preprocessing
from scipy.stats import norm
import  matplotlib.pyplot as plt
from mpmath.tests.test_quad import xtest_double_7
from dask.array.core import asarray
from numba.tests.test_array_methods import np_around_array
#from clase4.RegLin2 import pred



#obteniendo datos
myfilename = "./regLin.csv"
df = pd.read_csv(myfilename)
preprocessing.scale(df,copy=False)

x_train, x_test, y_train, y_test = train_test_split(df[df.columns[0:-1]],df[df.columns[-1]], train_size=0.75)
dfy = y_train[ ['y'] ].as_matrix() 
dfx = x_train[ ['X'] ].as_matrix()  

# OJO: len y shape no son lo mismo
#print dfx.shape
#print dfy.shape

xs = df[ ['X'] ] 
ys = df[ ['y'] ] 


#---Primera regresion
LR = LinearRegression()
LR.fit( x_train, y_train )
print "Resultados"
print  LR.intercept_,LR.coef_

#graficando
fig = plt.figure(figsize=(20,10))
ax1 = fig.add_subplot(1, 3, 1)
ax1.scatter(xs, ys, color='r')
#plt.show(block=False)

#calificando la calidad del modelo
pred = LR.predict(x_test)

errores = np.asmatrix(y_test)   - pred


ax2  = fig.add_subplot(1,3,2)
ax2.scatter(x_test, y_test, color='b')
ax2.plot(x_test, pred, color='blue',
         linewidth=3)


#mostrando la dependencia del error con la ordenada al origen
x0 = LR.intercept_
ws = []
es = []
for w in range(-50, +50):
    wi = x0 + w 
    temp  =  ( np.square (pred+wi-y_test) ).sum()
    ws.append(wi)
    es.append(temp)
    

ax3  = fig.add_subplot(1,3,3)
ax3.scatter(np.asmatrix( ws ) , np.asmatrix( es) , color='green')

plt.show()




print "<End of Program>"
