'''
Created on Sep 1, 2016

@author: Javier Quiroz
'''
#   Ejemplo usando transformacion sinuidal para xs
#
from sklearn.linear_model.base import LinearRegression
import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix
from scipy.stats import norm
import  matplotlib.pyplot as plt
from mpmath.tests.test_quad import xtest_double_7
from dask.array.core import asarray
from numba.tests.test_array_methods import np_around_array



#obteniendo datos
myfilename = "./regLin3.csv"
df = pd.read_csv(myfilename)
x_train, x_test, y_train, y_test = train_test_split(df[df.columns[0:-1]],df[df.columns[-1]], train_size=0.75)
dfy = y_train[ ['y'] ].as_matrix() #np.array( df['y']  ) no tiene el mismo efecto, espera una lista de columnas
dfx = x_train[ ['X'] ].as_matrix()# dnp.array( df['X']  )  
# OJO: len y shape no son lo mismo
#print dfx.shape
#print dfy.shape
#Transformar datos originales 
#xs = np.sin( df[ ['X'] ]  )
xs =  df[ ['X'] ]   
ys = df[ ['y'] ] 
x_train = np.sin(x_train)
x_test = np.sin(x_test )  


#---Primera regresion
LR = LinearRegression()
LR.fit( xs , ys )
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
print errores
print("Residual sum of squares: %.2f"
      % np.mean(errores  )** 2)
print LR.score(x_test, y_test)

ax2  = fig.add_subplot(1,3,2)
ax2.scatter(x_test, y_test, color='b')
ax2.plot(x_test, pred, color='blue',
         linewidth=3)
plt.show()
