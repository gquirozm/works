'''
Created on Sep 1, 2016

@author: javier
'''
# Operaciones con pandas
import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix
from scipy.stats import norm
import  matplotlib.pyplot as plt
from mpmath.tests.test_quad import xtest_double_7
from openpyxl.utils.units import dxa_to_cm


columns = ['spam', 'noSpam']
probs = list()
probs.append( 1) 

s = pd.Series([1,3,5,np.nan,6,8]) 
dates = pd.date_range('20130101', periods=6)
df = pd.DataFrame(np.random.randn(6,4), index=dates, columns=list('ABCD'))

#accesando a las dimensiones del dataframe
print df.shape
print df

#accesando a los compnentes
print df.index
print df.columns
print df.values

print df.describe()
print df.sort_index(axis=1, ascending=False)
print df.sort_values(by='B')
print df.loc[:,['A','B']]  # ojo una columna se selecciona usando un arreglo no un etiqueta

#selecting
print df.loc[:,['A','B']]  # ojo una columna se selecciona usando un arreglo o lista no un etiqueta
print df.loc['20130102':'20130104',['A','B']]
#obteniendo un escalar, las siguientes dos son equivalentes
print df.loc[dates[0],'A']
print df.at[dates[0],'A']
#seleccionando por posicion
dx = df.iloc[3] # devuelve una serie indexanda por los nombre de las columnas
print dx
print type(dx)
print dx.shape
print dx.T.shape # parece no tener efecto

print " *** aqui vamos "
print df.iloc[3:5,0:2]
print df.iloc[[1,2,4],[0,2]]
print type(df.iloc[[1,2,4],[0,2]]) # una manera rapida de escribir el type, En este caso un panda.Dataframe

print " *** slicing "


print df.iloc[1:3,:]
print df.iloc[:,1:3]
print type(df.iloc[:,1:3])

print df.iloc[1,1]  #un valor especifico
print df.iat[1,1]  #equivalente al anterior

#boolean indexing
print df[df.A > 0]
print df[df > 0] # muy interesante filter


#muy interesante
df2 = df.copy()
df2['E'] = ['one', 'one','two','three','four','three']
df2
print df2[df2['E'].isin(['two','four'])] #listando renglones especificos

#contnuar en lel articulo de 10 minuos con pandas en setting


print"termine"