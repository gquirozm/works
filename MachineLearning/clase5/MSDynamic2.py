'''
Created on Sep 22, 2016

@author: Javier Quiroz
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


w0_global = []
w1_global = []
rmse_global = []
myfilename = "./regLin.csv"
n = 0.5
x_train = pd.DataFrame()
x_test= pd.DataFrame()
y_train = pd.Series()
y_test = pd.Series()
wx = pd.Series()
ws_results = []
e_result = []
last_mse = 0.0
w0 = []
w1 = []
w2 = []
 
#=========================================================================
# Ejecuta un ciclo de ajuste linear dinamico
# Parametros: 
# xparam       - Xs a ajaustar, pandas datafrane
# yparam       - Ys a ajaustar, pandas series
# wparam       _ Ws a utilizar para inciar el calculo, pandas.Series
# nparam       - metaparametro eta del modelo, float
# ncolsparm    - numero de variables de xs, una variable por columna
# 
# Returns:
# tupla que contiene
# ws           - pesos ajustados,  pandas dataframe
# e            - ultimo error utilizado, float 
# es           - errores colectados en las iteraciones, pandas array
#
# colaterales: 1) espera reindexadas las estructuras usadas
#              2) utiliza arreglos globales para guardar ws
#=========================================================================
def lr_cycle(xparam, yparam, wparams, nparam,  ncolsparm , wparmsaux):

    # inicializando para aplicacion de algoritmo
    last_e = sys.float_info.max 
    
    xs = xparam
    ys = yparam
    last_w = wparams
    n = nparam
    
    print " ------------------------------------llego: last_w "
    print last_w
    print "id last_w= %i"%id(last_w)
    
    
    n_rows, n_cols = xs.shape
    n_cols = ncolsparm

    last_e = 0.0
    j = 0 # indexa columnas
    i = 0 # indexa renglones
    
    while ( i < n_rows):
        VxX = np.dot( xs.iloc[i, 0:n_cols] , last_w.T )
        last_e = float(  ys[i] - VxX  )
        for  j in range(n_cols):
            last_w[j] = float ( last_w[j] + n * last_e * xs.iloc[i,j] )
#         print " en ciclo interno"
#         print last_w
        wparmsaux.append(last_w)
        i  = i + 1
    
    w0_global.append(last_w[0])
    w1_global.append(last_w[1])
    t_rmse = rmse(xs, ys, last_w, n_cols)
    rmse_global.append(t_rmse)


    return (t_rmse)
         

#=========================================================================
# calcula el RMSE dada una serie ws
# Parametros: 
# xparam       - Xs a ajaustar, pandas datafrane
# yparam       - Ys a ajaustar, pandas series
# wparam       _ Ws a utilizar para inciar el calculo, pandas.Series
# ncolsparm    - numero de variables de xs, una variable por columna
# 
# Returns:
# tupla que contiene
# e            - ultimo error utilizado, float 
#
# colaterales: 
#=========================================================================
def rmse(xparam, yparam, wparams,  ncolsparm ):
    
    y_temp = wparams[0] * xparam["X0"] + wparams[1] * xparam["X"]
    RMSE = mean_squared_error(yparam, y_temp)**0.5
     
    return  RMSE
     
     
#=========================================================================
# Encuentra las ws mejores construidas en las n pasadas a observaciones
# Parametros: 

# Returns:
# tupla que contiene
# ws           - pesos ajustados que minimizan wse,  pandas dataframe
# best_rmse    - el mejor error raiz cuadrada medio
#
# colaterales: 1) utiliza arreglos globales para guardar ws
#              
#========================================================================
def find_best_rmse( ):  
    t_rmse =  pd.Series(rmse_global)
    indice_minimo = t_rmse.argmin() 
    best_rmse =t_rmse[indice_minimo] 
    w_best = [w0_global[indice_minimo], w1_global[indice_minimo]]
    
    print "indice minimo = ",indice_minimo 
    
    return ( w_best, best_rmse)
    
     
     


#=========================================================================
# Imprime ws y error colectados
# Parametros: 

# Returns:
# colaterales:
#              
#=========================================================================
def print_ws_and_errors():
    aux_size = len( w0_global ) 
    print "-----------Printing ws & error"
    for i in range(aux_size):
        print "w0    = %10f"% w0_global[i]
        print "w1    = %10f"% w1_global[i]
        print "error = %10f"% rmse_global[i]
    
#=========================================================================
# Plot
# Parametros: 
#
# Returns:
# colaterales:
#              
#=========================================================================
def plot_stuff():    
    # plt.axis([0, 2000, 0.22, 0.26])
    plt.xlabel("Iteracion")
    plt.ylabel("rmse")
    plt.title("RMSE")
    plt.plot(rmse_global)    
#plt.plot(e_result)
#plt.show()

#===========================================================================
#                             Main program
#===========================================================================
print "<Start of Program>"
#obteniendo datos
#myfilename = "./regLinx.csv"
df = pd.read_csv(myfilename)
preprocessing.scale(df,copy=False)  # escalando datos inplace
x_train, x_test, y_train, y_test = train_test_split(df[df.columns[0:-1]],df[df.columns[-1]], train_size=0.75, random_state = 0)

x_train['X0']  = 1
xs = x_train[ ['X0', 'X'] ]#[0:150] # por momento solo un conjunto pequeno de datos 
ys = y_train

xs.reset_index(drop=True, inplace=True);   #ojo con los indices que no llegan numerados de 0 a length-1
ys.reset_index(drop=True, inplace=True);


tempo = [1.0 for _ in xrange(1+1)]   #truco para generar el vector de ws con unos
wx = pd.Series(tempo)
n = 0.350
ws_results.append(wx)



for i in range(3):    
    print "Ciclo = ", i
    if ( i == 0 ):
        last_mse = lr_cycle(xs, ys, wx, n, 2,w0)
    if ( i == 1 ):
        last_mse = lr_cycle(xs, ys, wx, n, 2,w1)
    if ( i == 2 ):
        last_mse = lr_cycle(xs, ys, wx, n, 2,w2)   
    ws_results.append(wx)
    e_result.append(last_mse)
    #===========================================================================
    # print "---Resultados en ciclo---"
    # print '%8.7f  %8.6f' % (wx[0],  wx[1] ) 
    #===========================================================================
    #wx = twx 
    
#
#reporting
t_ws, t_rmse =  find_best_rmse()    
print "---Resultados---"
print t_ws
print "valor minimo rmse"
print t_rmse

print_ws_and_errors()
#plot_stuff()
print "******************** aqui vamos *******************"
print "imprimiento las ws de cada ciclo"
n_rows, n_cols = xs.shape
i = 0
for i in range (n_rows):
    print "w0 ---( %f  %f ) , w1 -- (%f  %f), w2 ---(%f  %f)"%(w0[i][0],w0[i][1], w1[i][0],w1[i][1], w2[i][0],w2[i][1])


print "<End of Program>"




