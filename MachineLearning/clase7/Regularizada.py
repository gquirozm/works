'''
Created on Sep 22, 2016

@author: javier
'''
'''
Created on Sep 8, 2016

@author: Javier Quiroz, Exercise for Machine Learning course, ITAM 2016
'''
from _cffi_backend import new_enum_type
'''
Created on Sep 1, 2016

@author: Javier Quiroz
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
from sklearn.linear_model.base import LinearRegression
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix
from scipy.stats import norm
from mpmath.tests.test_quad import xtest_double_7
from dask.array.core import asarray
from numba.tests.test_array_methods import np_around_array




#obteniendo datos
#myfilename = "./regLinx.csv"
myfilename = "./regLin.csv"
df = pd.read_csv(myfilename)
x_train, x_test, y_train, y_test = train_test_split(df[df.columns[0:-1]],df[df.columns[-1]], train_size=0.75)
dfy = y_train[ ['Y'] ].as_matrix() #np.array( df['y']  ) no tiene el mismo efecto, espera una lista de columnas
dfx = x_train[ ['X'] ].as_matrix()# dnp.array( df['X']  )  


df['X0']  = 1
xs = df[ ['X0', 'X'] ][1:150] # por momento solo un conjunto pequeno de datos 
ys = df[ ['y'] ][1:150]  



n = 0.089  



def ciclo(xparam, nparam ):
    e_temp = np.arange(0)
    # inicializando varlores para aplicacion de algoritmo
    last_e = sys.float_info.max
    xs = xparam
    n = nparam
    last_w = pd.Series([1,1], index=['X0',"X"])
    
    n_rows, n_cols = xs.shape
    j = 0 # indexa columnas
    i = 0 # indexa renglones
    
    while ( i < n_rows):
        suma = np.dot( xs.iloc[i] , last_w.T)
        VxX = suma
        last_e = ys.iloc[i,0] - VxX
        for  j in range(n_cols):
            if last_w.iloc[ j ] == 0: last_w.iloc[ j ] = 1
        new_w = last_w.T + n * last_e * xs.iloc[i]
        new_e = ys.iloc[i,0] - np.dot ( new_w.T ,  xs.iloc[i] )
        #print e_temp
        e_temp = np.append(e_temp, new_e)
     
        if ( np.abs(last_e) > np.abs(new_e) ) :
            print "encontre uno mejor"
            print last_e
            print new_e
            print last_w
            
            print new_w
            last_w = new_w
            last_e = new_e
           
        i  = i + 1
    #print e_temp 
    #test = np.array(e_temp)
    
    
    return (last_w, last_e, e_temp)
         
#print type(e_temp)
    
wx, we, temp = ciclo(xs, n)    
print "---Resultados---"
print wx
print "error final"
print we 
plt.plot(temp)
plt.show()
print temp
print type(temp)
print "End of Program>"




