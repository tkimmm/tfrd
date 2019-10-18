import pandas as pd
from matplotlib import pyplot as plt

x = [1,2,3]
y = [1,4,9]

# plt.plot(x, y)
# plt.title('whatever')
# plt.xlabel("x-axis")
# plt.ylabel("y-axis")
# plt.show()

data = pd.read_csv('countries.csv')
# us_data = data[data.country == 'United States']
# china_data = data[data.country == 'China']
# 
# plt.plot(us_data.year, us_data.population/10**6)
# plt.plot(us_data.year, china_data.population/10**6)
# plt.legend(['United States', 'China'])
# plt.xlabel('year')
# plt.ylabel('population')
# plt.show()

testdata = data[data.country == 'China']
print(testdata)
