from model import BayesMarkovModel
import numpy as np

# Define the size of the grid (e.g., 128x128 cells)
grid_size = 128

# Initialize the LGN layer with random activity (values 0 or 1 for inactive/active)
lgn_layer = np.random.choice([0, 1], size=(grid_size, grid_size))

# Initialize the SCI layer with zeros (inactive)
sci_layer = np.zeros((grid_size, grid_size))

# Parameters for the model
beta = 1.0  # Controls the influence of neighboring cells (as per the paper)
gamma = 4.0  # Weight for the SCI cell interactions
iterations = 10  # Number of iterations for the GICM algorithm

model = BayesMarkovModel(lgn_layer)

# Run the GICM algorithm for multiple iterations
for iteration in range(iterations):
    sci_layer = model.update_sci_layer(sci_layer, lgn_layer, beta, gamma)
