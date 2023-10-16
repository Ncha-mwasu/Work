import matplotlib.pyplot as plt
import numpy as np

X = ['10','20','30','40', '50', '60', '70', '80', '90', '100']
Ygirls = [10,20,20,40]
Zboys = [20,30,25,30]
#MAE
#y = [2.52E-05, 4.89E-05, 4.86E-05, 8.85E-05, 8.86E-05, 9.11E-05, 0.000122981, 9.97E-05, 0.000154072, 0.000132457]
#y2= [3E-5, 2.99E-05, 5.90597E-05, 0.002795799, 0.000106892, 0.000121161, 0.000164899, 0.000164899, 0.000150181, 0.000193922]

#pred_energy
y = [0.006571725, 0.006718875, 0.004197375, 0.0032754, 0.0060102, 0.002053125, 0.001419375, 0.001430175, 0.00142545, 0.00144165]
y2= [0.00364483125, 0.002958169, 0.00322005, 0.00279555, 0.003361181, 0.00279645, 0.002795119, 0.002937413, 0.003077738, 0.003361819]


X_axis = np.arange(len(X))

plt.bar(X_axis - 0.2, y, 0.4, color='r', label = 'Markov')
plt.bar(X_axis + 0.2, y2, 0.4, color = 'g', label = 'ARIMA')

plt.xticks(X_axis, X)
plt.xlabel("Prediction Intervals")
plt.ylabel("Average Energy (J)")
#plt.title("Number of Students in each group")
plt.legend()
plt.savefig('prediction_energy_plot.svg', dpi=300, bbox_inches='tight')
plt.show()
