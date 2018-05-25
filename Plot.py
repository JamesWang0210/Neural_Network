import matplotlib.pyplot as plt

# # 1.2.4
#
# x1 = [5, 20, 50, 100, 200]
# y1_1 = [0.535431222073, 0.143490816141, 0.0524036567181, 0.047132848027, 0.0469374509791]
# y1_2 = [0.705598679074, 0.556377195394, 0.48541936981, 0.43740992187, 0.440685272087]
#
# plt.title('Average Cross Entropy of the Data with Different Numbers of Hidden Units')
# plt.xlabel('Numbers of Hidden Units')
# plt.ylabel('Average Cross Entropy')
#
# plt.plot(x1, y1_1, color="red", linewidth=2.5, linestyle="-", label="Training Dataset")
# plt.plot(x1, y1_1, 'ro', color='black')
# plt.plot(x1, y1_2, color="blue", linewidth=2.5, linestyle="-", label="Validation Dataset")
# plt.plot(x1, y1_2, 'ro', color='black')
# plt.legend(loc='upper right')
# plt.axis([0, 250, 0, 0.75])
# plt.show()
#
# # 1.2.6
#
# x2 = [0.1, 0.01, 0.001]
# y2_1 = [0.0250880538617, 0.0524036567181, 0.383153297005]
# y2_2 = [0.835180016243, 0.48541936981, 0.522567099928]
#
# plt.title('Average Cross Entropy of the Data with Different Learning Rates')
# plt.xlabel('Learning Rate')
# plt.ylabel('Average Cross Entropy')
#
# plt.plot(x2, y2_1, color="red", linewidth=2.5, linestyle="-", label="Training Dataset")
# plt.plot(x2, y2_1, 'ro', color='black')
# plt.plot(x2, y2_2, color="blue", linewidth=2.5, linestyle="-", label="Validation Dataset")
# plt.plot(x2, y2_2, 'ro', color='black')
# plt.legend(loc='upper right')
# plt.axis([0, 0.12, 0, 1])
# plt.show()

# 1.2.6.1

x_3 = []
for i in range(1, 101):
    x_3.append(i)

# print len(x_3)

y_0_0 = map(float, open('train_0.txt').read().splitlines())
y_0_1 = map(float, open('valida_0.txt').read().splitlines())
y_1_0 = map(float, open('train_1.txt').read().splitlines())
y_1_1 = map(float, open('valida_1.txt').read().splitlines())
y_2_0 = map(float, open('train_2.txt').read().splitlines())
y_2_1 = map(float, open('valida_2.txt').read().splitlines())

plt.title('Average Cross Entropy of the Data with Different Learning Rates')
plt.xlabel('Number of Epoch')
plt.ylabel('Average Cross Entropy')

plt.plot(x_3, y_0_0, color="blue", linewidth=2.5, linestyle="-", label="Training Dataset with Lambda=0.1")
plt.plot(x_3, y_0_1, color="blue", linewidth=2.5, linestyle="--", label="Validation Dataset with Lambda=0.1")
plt.plot(x_3, y_1_0, color="green", linewidth=2.5, linestyle="-", label="Training Dataset with Lambda=0.01")
plt.plot(x_3, y_1_1, color="green", linewidth=2.5, linestyle="--", label="Validation Dataset with Lambda=0.01")
plt.plot(x_3, y_2_0, color="purple", linewidth=2.5, linestyle="-", label="Training Dataset with Lambda=0.001")
plt.plot(x_3, y_2_1, color="purple", linewidth=2.5, linestyle="--", label="Validation Dataset with Lambda=0.001")
# plt.plot(x2, y2_1, 'ro', color='black')
# plt.plot(x2, y2_2, color="blue", linewidth=2.5, linestyle="-", label="Validation Dataset")
# plt.plot(x2, y2_2, 'ro', color='black')
plt.legend(loc='upper right')
plt.axis([0, 100, 0, 2.3])
plt.show()
