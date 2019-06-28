import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

correl_list = []
x_axis = []

for i in range(10,91,10) :
    
    x_axis.append(i)
    correl_list.append(pd.read_csv('results1_csv/output_results_test' + str(i) + '.csv' ).iloc[0, 3])


print(correl_list)

plt.title('DAPL Prediction')
plt.xlabel('Missing Percentage')
plt.ylabel('Correlation')
plt.plot(x_axis,correl_list)
plt.show()
