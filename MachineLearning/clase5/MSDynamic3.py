'''
Created on Oct 6, 2016

@author: javier

Version with approach pseudo objects

'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
from _cffi_backend import new_enum_type
from sklearn.linear_model.base import LinearRegression
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import mean_squared_error
from sklearn import preprocessing
from scipy.stats import norm
from mpmath.tests.test_quad import xtest_double_7
from dask.array.core import asarray
from numba.tests.test_array_methods import np_around_array
from sklearn.decomposition.tests.test_nmf import random_state


# object variables
w0= []
w1= []
rmse = []
myfilename = "./regLin.csv"
n = 0.0
x_train = pd.DataFrame()
x_test= pd.DataFrame()
y_train = pd.Series()
y_test = pd.Series()
xs = pd.DataFrame()
ys = pd.Series()
wx = pd.Series()
num_rows = 0
num_cols = 0
last_w = []


def init_vars(eta,num_vars):
    df = pd.read_csv(myfilename)
    preprocessing.scale(df,copy=False)  # escalando datos inplace
    lx_train, lx_test, ly_train, ly_test = train_test_split(df[df.columns[0:-1]],df[df.columns[-1]], train_size=0.75, random_state = 0)

    lx_train['X0']  = 1
    lxs = lx_train[ ['X0', 'X'] ]#[0:150] # por momento solo un conjunto pequeno de datos 
    ys = pd.Series(ly_train)
    xs = lxs.copy(deep=True)
    

    xs.reset_index(drop=True, inplace=True);   #ojo con los indices que no llegan numerados de 0 a length-1
    ys.reset_index(drop=True, inplace=True);

    num_rows, num_cols = xs.shape

    for i in range(num_rows):
        last_w.append(1.0)
    n = eta

    
    
def learn():
    
    last_e = 0.0
    j = 0 # indexa columnas
    i = 0 # indexa renglones
    
    while ( i < num_rows):
        VxX = np.dot( xs.iloc[i, 0:num_cols] , last_w.T )
        last_e = float(  ys[i] - VxX  )
        for  j in range(num_cols):
            last_w[j] = float ( last_w[j] + n * last_e * xs.iloc[i,j] )
        i  = i + 1
    
    w0.append(last_w[0])
    w1.append(last_w[1])
    t_rmse = rmse()
    rmse.append(t_rmse)

    return (t_rmse)
         

def rmse():
    y_temp = last_w[0] * xs["X0"] + last_w[1] * xs["X"]
    RMSE = mean_squared_error(ys, y_temp)**0.5
     
    return  RMSE


def find_best_rmse( ):  
    t_rmse =  pd.Series(rmse)
    indice_minimo = t_rmse.argmin() 
    best_rmse =t_rmse[indice_minimo] 
    w_best = [w0[indice_minimo], w1[indice_minimo]]
    
    print "indice minimo = %i"%(indice_minimo) 
    
    return ( w_best, best_rmse)


def print_ws_and_errors():
    aux_size = len( w0 ) 
    print "-----------Printing ws & error"
    for i in range(aux_size):
        print "w0    = %10f"% w0[i]
        print "w1    = %10f"% w1[i]
        print "error = %10f"% rmse[i]
    
def plot_stuff():    
    # plt.axis([0, 2000, 0.22, 0.26])
    plt.xlabel("Iteracion")
    plt.ylabel("rmse")
    plt.title("RMSE")
    plt.plot(rmse)    
#plt.plot(e_result)
#plt.show()


#===========================================================================
#                             Main program
#===========================================================================
print "<Start of Program>"

init_vars(eta= 0.35, num_vars=2)

for i in range(3):    
    print "Ciclo = ", i
    learn()

#reporting
t_ws, t_rmse =  find_best_rmse()    
print "---Resultados---"
print "Best results w0 = %f, w1=%f "%(t_ws[0], t_ws[1])
print "valor minimo rmse"
print t_rmse

print_ws_and_errors()
#plot_stuff()


print "<End of Program>"
    
    
    
