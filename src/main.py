import ONE_HL_NN as OHL
import numpy as np
import matplotlib.pyplot as plt

print("--- Starting Run One ---")
nn = OHL.NeuralNet()
nn.load_data()
nn.train()
print("--- Run One Over ---")

print("--- Starting Run Two ---")
nn.learning_rate = 0.00001
nn.iterations = 100000
nn.train()
print("--- Run Two Over ---")

print("--- Final parameters ---")
print(nn.parameters)

wide_y_predicted = []
for x in nn.norm_domain:
    p_x, _, _ = nn.forward_pass(x)
    wide_y_predicted.append(p_x)

x_points = np.linspace(min(nn.domain), max(nn.domain), 100)
plot_x_normalized = nn.normalize(x_points)

plot_y_predicted = []
for x_norm in plot_x_normalized:
    p_x, _, _ = nn.forward_pass(x_norm)
    plot_y_predicted.append(p_x)

# Plot 1: The 100-point predicted curve
plt.plot(x_points, plot_y_predicted, color='red', label='Model Prediction')

# Plot 2: The 7 given data points
plt.scatter(nn.domain, nn.y_data, color='blue', label='Actual Data')

# Labeling
plt.title("Neural Network Fit (" + str(nn.neurons) + " neurons)")
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
# Shows labels for plots
plt.legend()
# Adds a grid
plt.grid(True)

# Shows the plot
plt.show()
