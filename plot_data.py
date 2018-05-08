import pickle
import matplotlib.pyplot as plt
import statistics as stat

speaking_data = pickle.load(open('data/data2.p', 'rb'))
non_speaking_data = pickle.load(open('data/data1.p', 'rb'))

time1 = [t for t in range(len(speaking_data))]
time2 = [t for t in range(len(non_speaking_data))]


max_s, min_s = max(speaking_data), min(speaking_data)
speaking_data = [(d - min_s) / (max_s - min_s) for d in speaking_data]
max_ns, min_ns = max(non_speaking_data), min(non_speaking_data)
non_speaking_data = [(d - min_ns) / (max_ns - min_ns) for d in non_speaking_data]


plt.plot(time1, speaking_data)
plt.plot(time2, non_speaking_data)
plt.show()

print(stat.stdev(speaking_data), stat.stdev(non_speaking_data))