'''
Created on Oct 6, 2016

@author: javier
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

class Adjuster:

    def __init__( self, eta ):
        self.x_train = pd.DataFrame()
        self.y_train = pd.DataFrame()
        self.x_test = pd.DataFrame()
        self.y_test = pd.DataFrame()
        self.xs = pd.DataFrame()
        self.ys = pd.Series()
        self.n = eta
        self.num_rows = 0
        self.num_cols =0
        self.last_w = []
        self.w0= []
        self.w1= []
        self.rmse = []
        
    def read_data( self ):
        myfilename = "./regLin.csv"
        df = pd.read_csv(myfilename)
        preprocessing.scale(df,copy=False)  # escalando datos inplace
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(df[df.columns[0:-1]],df[df.columns[-1]], train_size=0.75, random_state = 0)

        self.x_train['X0']  = 1
        self.xs = self.x_train[ ['X0', 'X'] ][0:750] # por momento solo un conjunto pequeno de datos 
        self.ys = self.y_train[0:750]
        self.xs.reset_index(drop=True, inplace=True); #ojo con los indices que no llegan numerados de 0 a length-1
        self.ys.reset_index(drop=True, inplace=True);

        self.num_rows, self.num_cols = self.xs.shape

        for i in range(self.num_cols):
            self.last_w.append(1.0)
        
        
    def learn( self ):
        temp_rmse = 0.0
        last_e = 0.0
        j = 0 # indexa columnas
        i = 0 # indexa renglones
    
        while ( i < self.num_rows):
            t_last_w = pd.Series(self.last_w)
            #print self.xs.iloc[i]
            #print t_last_w
            VxX = np.dot( self.xs.iloc[i] , t_last_w)
            #print "printing ys"%(self.ys)
            last_e = float( self.ys[i] - VxX  )
            for  j in range(self.num_cols):
                self.last_w[j] = float ( self.last_w[j] + self.n * last_e * self.xs.iloc[i,j] )
                i  = i + 1
        self.w0.append(self.last_w[0])
        self.w1.append(self.last_w[1])
        temp_rmse = self.calculate_rmse()
        self.rmse.append( temp_rmse )
        
        return  temp_rmse 
         

    def calculate_rmse(self):
        y_temp = self.last_w[0] * self.xs["X0"] + self.last_w[1] * self.xs["X"]
        RMSE = mean_squared_error(self.ys, y_temp)**0.5
     
        return  RMSE


    def find_best_rmse( self ):  
        t_rmse =  pd.Series(self.rmse)
        indice_minimo = t_rmse.argmin() 
        best_rmse =t_rmse[indice_minimo] 
        w_best = [self.w0[indice_minimo], self.w1[indice_minimo]]
    
        print "indice minimo = %i"%(indice_minimo) 
        return ( w_best, best_rmse)


    def print_ws_and_errors( self ):
        aux_size = len( self.w0 ) 
        print "-----------Printing ws & error"
        for i in range(aux_size):
            print "w0 = %10f,w1= %10f, error = %10f" % ( self.w0[i], self.w1[i], self.rmse[i])
            
    def plot_stuff( self ):    
    # plt.axis([0, 2000, 0.22, 0.26])
        plt.xlabel("Iteracion")
        plt.ylabel("rmse")
        plt.title("RMSE")
        plt.plot(self.rmse)    
#plt.plot(e_result)
        plt.show()
        
    def print_xs(self):
        print "printing xs"
        print self.x_train
    


#===========================================================================
#                             Main program
#===========================================================================
print "<Start of Program>"

myAdjuster = Adjuster(eta = 0.0179)
myAdjuster.read_data()
#myAdjuster.print_xs()


for i in range(10):    
    print "Ciclo = ", i
    myAdjuster.learn()

#reporting
t_ws, t_rmse =  myAdjuster.find_best_rmse()    
print "---Resultados---"
print "Best results w0 = %f, w1=%f "%(t_ws[0], t_ws[1])
print "valor minimo rmse"
print t_rmse

myAdjuster.print_ws_and_errors()
myAdjuster.plot_stuff()


print "<End of Program>"
    