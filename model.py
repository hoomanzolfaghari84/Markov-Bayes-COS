import numpy as np

class BayesMarkovModel():
    def __init__(self, grid_size):
        self.grid_size = grid_size

    # Function to calculate clique energy for a given neighborhood
    def clique_energy(self, cell, neighbors, beta, horizontal=True):
        energy = 0
        if horizontal:
            for neighbor in neighbors:
                energy += -beta if cell == neighbor else beta
        else:
            for neighbor in neighbors:
                energy += beta if cell == neighbor else -beta
        return energy


    # Define the neighborhood structure (3x3 grid around each cell)
    def get_neighbors(self, grid, i, j):
        neighbors = []
        for x in range(max(0, i - 1), min(self.grid_size, i + 2)):
            for y in range(max(0, j - 1), min(self.grid_size, j + 2)):
                if (x, y) != (i, j):
                    neighbors.append(grid[x, y])
        return neighbors

    # Function to update the SCI layer using GICM
    def update_sci_layer(self, sci_layer, lgn_layer, beta, gamma):
        new_sci_layer = np.copy(sci_layer)
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                # Get the neighbors of the current cell
                neighbors = self.get_neighbors(sci_layer, i, j)
                
                # Calculate clique energies
                horizontal_energy = self.clique_energy(sci_layer[i, j], neighbors, beta, horizontal=True)
                non_horizontal_energy = self.clique_energy(sci_layer[i, j], neighbors, beta, horizontal=False)
                
                # Update the SCI cell based on the LGN input and neighboring SCI cells
                if lgn_layer[i, j] == 1:  # Active LGN cell
                    new_sci_layer[i, j] = 1 if horizontal_energy < non_horizontal_energy else 0
                else:
                    new_sci_layer[i, j] = 0  # Inactive LGN cell
        return new_sci_layer

