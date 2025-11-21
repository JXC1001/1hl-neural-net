import math
import random

class NeuralNet:

    def __init__(self):
        print("Initializing NeuralNet...")
        self.parameters = {}
        self.neurons = 10
        self.loss = 0
        self.learning_rate = 0.01
        self.relu_alpha = 0.01
        self.iterations = 100000
        self.norm_domain = []
        self.domain = []
        self.y_data = []
        self.p_i = []
        print("Done!")

    def load_data(self):
        ## Define neuron amount
        self.neurons = int(input("How many neurons for the network? "))

        ## Parameter initialization
        parameters = {
            "w_h": [],
            "b_h": [],
            "w_o": [],
            "b_o": None
        }

        for key in parameters.keys():
            if key == "b_o":
                parameters[key] = random.uniform(-0.1, 0.1)
            else:
                param_values = []
                for _ in range(self.neurons):
                    param_values.append(random.uniform(-0.1, 0.1))
                parameters[key] = param_values
        self.parameters = parameters
        ## x and y values loading
        d_len = int(input("How big is the domain of this neural network? "))
        centered = "n"
        if d_len % 2 == 1:
            centered = input("Do you wish to center the domain at 0 evenly? (answer 'y' or 'n') ")
        
        if centered == "y":
            starting = (d_len - 1) / 2
            for i in range(d_len):
                self.domain.append(-starting + i)
        elif centered == "n":
            for i in range(d_len):
                self.domain.append(float(input("What's the corresponding x-value for slot " + str(i + 1) + "? ")))
        else:
            print("-- Invalid input, try again --")
            centered = input("Do you wish to center the domain at 0 evenly? (answer 'y' or 'n') ")
        
        for value in self.domain:
                self.y_data.append(float(input("What's the corresponding y-value at x = " + str(value) + "? ")))
        ## Learning/model hyperparameters
        self.learning_rate = float(input("What is the learning rate? "))
        self.relu_alpha = float(input("What is ReLU's leakiness? "))
        self.iterations = int(input("How many iterations? "))
        ## Normalize before letting the user do something else
        self.norm_domain = self.normalize(domain=self.domain)

    @staticmethod
    def normalize(domain):
        x_min = min(domain)
        x_max = max(domain)
        x_norm = []
        for x in domain:
            x_norm.append(2 * ((x - x_min) / (x_max - x_min)) - 1)
        return x_norm

    def forward_pass(self, current_x):
        z_hidden = []
        h_hidden = []
        p_x = 0

        for j in range(self.neurons):
            ## -- Hidden Layer --
            w_h = self.parameters["w_h"][j]
            b_h = self.parameters["b_h"][j]
            z = (w_h * current_x) + b_h
            z_hidden.append(z)
            h_hidden.append(max(self.relu_alpha * z, z))

            ## -- Output Layer --
            w_o = self.parameters["w_o"][j]
            p_x += w_o * h_hidden[j]
        p_x += self.parameters["b_o"]
        return p_x, z_hidden, h_hidden

    def backward_pass(self):
        gradients = {
            "w_h": [0] * self.neurons,
            "b_h": [0] * self.neurons,
            "w_o": [0] * self.neurons,
            "b_o": 0
        }

        ## Reset p_i to avoid infinite element appending
        self.p_i = []

        for i in range(len(self.y_data)):
            ## Execute Forward Step and save values
            x_value = self.norm_domain[i]
            p_x, z_hidden_x, h_hidden_x = self.forward_pass(x_value)
            ## Save predicted values (used later for loss calculation)
            self.p_i.append(p_x)

            ## Execute backward step
            y_actual = self.y_data[i]
            ## Chain rule for SQE: (y-actual - y-predicted)^2
            ## This gets us the same base for every partial derivative
            grad_base = -2 * (y_actual - p_x)

            ## Update gradients
            gradients["b_o"] += grad_base

            for neuron in range(self.neurons):

                gradients["w_o"][neuron] += grad_base * h_hidden_x[neuron]

                if z_hidden_x[neuron] > 0:
                    gradients["w_h"][neuron] += grad_base * self.parameters["w_o"][neuron] * x_value
                    gradients["b_h"][neuron] += grad_base * self.parameters["w_o"][neuron]
                else:
                    gradients["w_h"][neuron] += grad_base * self.relu_alpha * self.parameters["w_o"][neuron] * x_value
                    gradients["b_h"][neuron] += grad_base * self.relu_alpha * self.parameters["w_o"][neuron]
        return gradients

    ## Updates parameters using the new gradients
    def update_parameters(self, gradients):
        self.parameters["b_o"] -= self.learning_rate * gradients["b_o"]

        for neuron in range(self.neurons):
            self.parameters["w_o"][neuron] -= self.learning_rate * gradients["w_o"][neuron]
            self.parameters["b_h"][neuron] -= self.learning_rate * gradients["b_h"][neuron]
            self.parameters["w_h"][neuron] -= self.learning_rate * gradients["w_h"][neuron]

    ## Calculates loss using the saved y-predicted values saved in
    ## p_i in the backward pass
    def calculate_loss(self):
        self.loss = 0
        for i in range(len(self.y_data)):
            actual_y = self.y_data[i]
            predicted_y = self.p_i[i]
            self.loss += math.pow(actual_y - predicted_y, 2)
        return self.loss

    ## The main training process
    def train(self):
        for iteration in range(self.iterations):
            derivatives = self.backward_pass()
            if iteration % 10000 == 0:
                loss = self.calculate_loss()
                print("Loss: " + str(loss))
            self.update_parameters(derivatives)
