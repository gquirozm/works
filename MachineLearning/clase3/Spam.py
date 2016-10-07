'''
Created on Aug 25, 2016

@author: Javier Quiroz
'''
import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix
from scipy.stats import norm
import  matplotlib.pyplot as plt
from mpmath.tests.test_quad import xtest_double_7


def califica(pSpam, pNoSpam, Y):
    print "in function lists"
    print pSpam
    print pNoSpam
    print (type(Y))
    #===========================================================================
    # for name, val in Y.index:
    #     print name
    #===========================================================================
    n = len( pSpam )
    print Y.index
    print Y.values
    acertados = 0 
    fallados = 0 

    for i in range(0,n)  :
        if ( pSpam [ i ] > pNoSpam [ i ] ) :
            if (Y [ i ] == 1 ) :
                acertados  = acertados + 1 
            else:
                fallados = fallados + 1
        else:
            if (Y [ i ] == 0 ) :
                acertados  = acertados + 1
            else:
                fallados = fallados + 1
    total = n
    return ( acertados, fallados, total)
            
            
            


df = pd.read_csv("/Users/javier/Documents/2016ITAMOtono/MachineLearning/clase3/spambase.data", header = None)
# dividiendo los datos en entrenamiento  y prueba 
X_train, X_test, Y_train, Y_test = train_test_split(df[df.columns[0:-1]],df[df.columns[-1]], train_size=0.75)

#calculamos loa medias y sd de cada una de las variables 
X_train_mean_spam = X_train[Y_train==1].mean()
X_train_mean_nospam = X_train[Y_train==0].mean()
X_train_sd_spam = X_train[Y_train==1].std()
X_train_sd_nospam = X_train[Y_train==0].std()
positivos = negativos = totales = 0
for e in Y_train   :
    if (e == 1) : positivos += 1
    else: negativos +=1
totales = positivos + negativos
pSpam = positivos* (1.0) / totales 
pNoSpam = negativos* (1.0)/totales 
print "positivos= %d"% (positivos )
print "negativos= %d"% (negativos )
print "totales= %d"% (totales )
print "pSpam"
print pSpam
print "pNoSpam"
print pNoSpam

#recuperamos columnas para calculo
nColumns = len (X_test.columns)
nRows = len(X_test.index)
print nColumns
print nRows
# dos listas para guardar probabiliadades
probsSpam = []
probsNoSpam = []

print ("comenzando a calificar")
#procedemos a probar el modelo con grupo de datos Test
#truco para pararlo
nRows = 1
for i in range( 0, nRows ):
    sumaSpam = 0.0
    sumaNoSpam = 0.0
    for j in range( 0,nColumns ):
        print "i= %d j= %d" % ( i, j )
        sumaSpam = sumaSpam + np.log( norm( X_train_mean_spam.iloc[j],  X_train_sd_spam.iloc[ j ] ).pdf( X_test.iloc[ i, j ] ) )
        sumaNoSpam = sumaNoSpam + np.log( norm( X_train_mean_nospam.iloc[j],  X_train_sd_nospam.iloc[ j ] ).pdf( X_test.iloc[ i, j ] ))
        print "sumaSpam = %f" % ( sumaSpam )
    probsSpam.append( sumaSpam + np.log( pSpam ))
    probsNoSpam.append( sumaNoSpam+np.log( pNoSpam ))
    

print len( probsSpam ) 
print len( probsNoSpam ) 

print ( probsSpam ) 
print ( probsNoSpam ) 
Y_test.reindex()
acertados, fallados, totales  = califica(probsSpam, probsNoSpam, Y_test)
print( "---------Resutados-----------")
a = "Acertados = "+  repr( acertados ) 
f = "Fallados= " + repr( fallados ) 
t = "Total = ", + repr( totales ) 
print( a )
print( f )
print( t )

#X_train_mean_spam es un vector que contiene las medias de cada variable


              
        
