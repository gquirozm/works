import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv('/Users/javier/Documents/2016ITAMOtono/MachineLearning/DataForPandas.csv')
#aqui se crea una columna
df['bono'] = df.saldo * 0.25
#df.bono
#print df
#aqui se cuentan renglones
#(unos, dos, tres, cuatro,cinco) 
unos, dos, tres, cuatro,cinco = df[df.contador==1].sum()
print ("aqui va el contador" , cuatro)