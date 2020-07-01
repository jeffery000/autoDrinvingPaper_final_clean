from matplotlib import pyplot as plt
f1 = open("save_data_first.txt","r").readlines()
f2 = open("save_data_second.txt","r").readlines()
f3 = open("save_data_third.txt","r").readlines()
first_x = [float(i.split()[6].split(",")[0][1:]) for i in f1]
first_y = [float(i.split()[6].split(",")[1][:-1]) for i in f1]
second_x = [float(i.split()[6].split(",")[0][1:]) for i in f2]
second_y = [float(i.split()[6].split(",")[1][:-1]) for i in f2]
third_x = [float(i.split()[6].split(",")[0][1:]) for i in f3]
third_y = [float(i.split()[6].split(",")[1][:-1]) for i in f3]
plt.plot(first_x,first_y,"--",c="b")
plt.plot(second_x,second_y,"--",c="g")
plt.plot(third_x,third_y,"--",c="r")
plt.show()