import sys
import numpy as np

train_in = sys.argv[1]
val_in = sys.argv[2]
train_out = sys.argv[3]
val_out = sys.argv[4]
met_out = sys.argv[5]
epoch = int(sys.argv[6])  # Number of process to go through all the data
hid = int(sys.argv[7])  # Number of Hidden Units
flag = int(sys.argv[8])  # Initial Mode: 1 or 2
rate = float(sys.argv[9])  # Learning Rate for SGD

nr_x = 129
nr_y = 10


class Neural:
    def __init__(self):
        self.filename = ''
        self.D = np.zeros((1, 1))  # Get Raw Data/Features
        self.Y = np.zeros((1, 1))  # Get Raw Labels
        self.alpha = np.zeros((hid, nr_x))  # Alpha Weights
        self.beta = np.zeros((nr_y, hid+1))  # Beta Weights
        self.a = np.zeros((hid, 1))  # All 'a' for Sigmoid
        self.I = np.ones((1, hid))  # I for Sigmoid
        self.z = np.zeros((hid+1, 1))  # All Hidden Values 'z'
        self.b = np.zeros((nr_y, 1))  # All 'b' for Softmax
        self.y_hat = np.zeros((nr_y, 1))  # Probabilities of different y
        self.y = np.zeros((nr_y, 1))  # One-hot Code of Each Data
        self.J = 0  # Cross Entropy
        self.dJdy = np.zeros((nr_y, 1))  # dJ/dy_hat
        self.dJdb = np.zeros((nr_y, nr_y))  # dJ/db
        self.dJd_beta = np.zeros((nr_y, hid+1))  # dJ/d_beta
        self.dJdz = np.zeros((hid, 1))  # dJ/dz
        self.dJda = np.zeros((hid, 1))  # dJ/da
        self.dJd_alpha = np.zeros((hid, nr_x))  # dJ/d_alpha
        self.wrong = 0

    def get(self, fn):
        self.filename = fn
        self.D = np.genfromtxt(self.filename, dtype=None, delimiter=',')
        self.D[:, 0] = 1
        self.Y = np.genfromtxt(self.filename, dtype=None, delimiter=',', usecols=(0))  # Column Vector

    def initial(self):  # Initial All the Weights
        if flag == 1:
            self.alpha = np.random.uniform(-0.1, 0.1, (hid, nr_x))
            self.beta = np.random.uniform(-0.1, 0.1, (nr_y, hid+1))
            self.alpha[:, 0] = 0
            self.beta[:, 0] = 0

    def linear_for(self, a, b):
        c = np.dot(a, b)
        return c

    def sigmoid_for(self):
        self.z = np.divide(self.I, (self.I + np.exp((-1)*self.a)))

    def soft_for(self):
        e = np.exp(self.b)
        self.y_hat = np.divide(e, sum(e))

    def cross_for(self):
        J = - np.dot(np.transpose(self.y), np.log(self.y_hat))[0]
        return J

    def linear_back(self, a, b):  # Compute Weights
        c = np.dot(a, b)
        return c

    def sigmoid_back(self):
        self.dJda = self.dJdz * self.z * (np.transpose(self.I)-self.z)

    def soft_back(self):
        b_ca = np.concatenate((self.y_hat, self.y_hat, self.y_hat, self.y_hat, self.y_hat, self.y_hat, self.y_hat,
                               self.y_hat, self.y_hat, self.y_hat,), axis=1)
        diag = np.diag(np.diag(b_ca))  # Matrix with "y_hat" diagnosis
        dydb = (diag - np.dot(self.y_hat, np.transpose(self.y_hat)))  # dy/db
        self.dJdb = np.transpose(np.dot(np.transpose(self.dJdy), dydb))  # dJ/db

    def cross_back(self):
        self.dJdy = (-1) * np.divide(self.y, self.y_hat)

    def forward(self, d):  # Forward
        self.a = self.linear_for(self.alpha, d)
        self.sigmoid_for()
        self.z = np.transpose(np.insert(self.z, 0, 1, axis=1))  # Add bias term
        self.b = self.linear_for(self.beta, self.z)
        self.soft_for()

    def backward(self, d):  # Backpropagation
        self.cross_back()
        self.soft_back()
        self.dJd_beta = self.linear_back(self.dJdb, np.transpose(self.z))
        self.dJdz = np.transpose(self.linear_back(np.transpose(self.dJdb), self.beta))
        self.dJdz = np.delete(self.dJdz, 0, axis=0)
        self.z = np.delete(self.z, 0, axis=0)
        self.sigmoid_back()
        d = np.reshape(d, (1, nr_x))
        self.dJd_alpha = self.linear_back(self.dJda, d)

    def learning(self):
        for i in range(0, len(self.D)):
            self.y[self.Y[i]] = 1  # Create One-hot Code for Each Data
            self.forward(self.D[i])
            self.backward(self.D[i])
            self.alpha -= rate*self.dJd_alpha  # Update all the weights
            self.beta -= rate*self.dJd_beta
            self.y = np.zeros((nr_y, 1))

    def cross_entropy(self, e):
        for i in range(0, len(self.D)):
            self.y[self.Y[i]] = 1  # Create One-hot Code for Each Data
            self.forward(self.D[i])
            self.J += float(self.cross_for())/len(self.D)
            self.y = np.zeros((nr_y, 1))

        if e == 1 and self.filename == train_in:
            f = open(met_out, 'w')
        else:
            f = open(met_out, 'a')

        if self.filename == train_in:
            f.write("epoch=" + str(e) + " crossentropy(train): " + str(self.J) + '\n')
        else:
            f.write("epoch=" + str(e) + " crossentropy(validation): " + str(self.J) + '\n')

        self.J = 0
        f.close()

    def error(self):
        for i in range(0, len(self.D)):
            self.y[self.Y[i]] = 1  # Create One-hot Code for Each Data
            self.forward(self.D[i])
            max = np.argmax(self.y_hat)
            if max != self.Y[i]:
                self.wrong += 1

            if self.filename == train_in:
                if i == 0:
                    f1 = open(train_out, "w")
                else:
                    f1 = open(train_out, "a")
            else:
                if i == 0:
                    f1 = open(val_out, "w")
                else:
                    f1 = open(val_out, "a")

            f1.write(str(max) + '\n')
            f1.close()

            self.y = np.zeros((nr_y, 1))

        Error = float(self.wrong)/len(self.D)
        f2 = open(met_out, 'a')
        if self.filename == train_in:
            f2.write("error(train): " + str(Error) + '\n')
        else:
            f2.write("error(validation): " + str(Error) + '\n')
        f2.close()

tr = Neural()
va = Neural()

tr.get(train_in)
va.get(val_in)

tr.initial()

for e in range(1, epoch+1):
    tr.learning()

    va.alpha = tr.alpha
    va.beta = tr.beta

    tr.cross_entropy(e)
    va.cross_entropy(e)

tr.error()
va.error()

