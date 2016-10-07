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

class myclase:

    def __init__(self):
        ##self.x_train = pd.DataFrame()
        self.y_train = pd.DataFrame()
        self.x_test = pd.DataFrame()
        self.y_test = pd.DataFrame()
        self.xs = pd.DataFrame()
        print "Ejecute el constructor"
        
    def change_df(self):
        myfilename = "./regLin.csv"
        df = pd.read_csv(myfilename)
        preprocessing.scale(df,copy=False)  # escalando datos inplace
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(df[df.columns[0:-1]],df[df.columns[-1]], train_size=0.75, random_state = 0)

        self.x_train['X0']  = 1
        self.xs = self.x_train[ ['X0', 'X'] ][0:200] # por momento solo un conjunto pequeno de datos 
        
    def print_xs(self):
        print "printing xs"
        print self.x_train
        
    
        
        
############# Programa principal

unaclase = myclase()
unaclase.change_df()
unaclase.print_xs()
    