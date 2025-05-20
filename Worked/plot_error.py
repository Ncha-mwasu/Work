import sklearn
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.interpolate import make_interp_spline
#mse
#y = np.array([3.03957E-08, 8.78E-08, 5.76E-08, 2.89E-07, 2.64E-07, 2.65E-07, 6.53E-07, 5.36E-07, 1.66E-06, 1.16E-06])
#y2= np.array([1.76E-07, 1.31E-09, 4.22E-09, 0.000452001, 6.81003E-08, 3.63675E-08, 3.6598E-08, 3.66E-08, 3.16308E-08, 5.31194E-08])
#MAE
y = np.array([2.52E-05, 4.89E-05, 4.86E-05, 8.85E-05, 8.86E-05, 9.11E-05, 0.000122981, 9.97E-05, 0.000154072, 0.000132457])
y2= np.array([0.00020724941002276388, 2.99E-05, 5.90597E-05, 0.002795799, 0.000106892, 0.000121161, 0.000164899, 0.000164899, 0.000150181, 0.000193922])

x = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
#correlation = y.corr(x)
#print (correlation)
xnew = np.linspace(x.min(), x.max(), 200)
gfg = make_interp_spline(x, y, k=2)
grg = make_interp_spline(x, y2, k=3)
  
y_new = gfg(xnew)
y2_new = grg(xnew)
# adds the title
#plt.title('MAE_MARKOV_CHAIN')

# plotting the data
plt.plot(xnew, y_new, color = 'r', label = 'Markov')
plt.plot(xnew, y2_new, color = 'g', label = 'ARIMA')

# This will fit the best line into the graph
#plt.plot(np.unique(x), np.poly1d(np.polyfit(x, y, 1))
#		(np.unique(x)), color='red')
plt.legend()
# Labelling axes
plt.xlabel('Prediction Interval')
plt.ylabel('Average MAE')
plt.savefig('MAE_plot.svg', dpi=300, bbox_inches='tight')
plt.show()