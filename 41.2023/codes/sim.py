import numpy as np
import matplotlib.pyplot as plt

# Parameters
num_simulations = 10000  # Number of simulations

# Initialize variables
count_U_le_half = 0
simulation_results = []

# Simulating the scenario
for _ in range(num_simulations):
    # Generating random vectors from a multivariate normal distribution
    X = np.random.multivariate_normal(mean=[0, 0], cov=[[1, 0.5], [0.5, 1]], size=10)
    sample_mean = np.mean(X, axis=0)
    mu = np.array([0, 0])
    Sigma = np.array([[1, 0.5], [0.5, 1]])
    inverse_Sigma = np.linalg.inv(Sigma)
    
    # Calculating U
    U = 1 / (1 + np.dot(np.dot((sample_mean - mu).T, inverse_Sigma), (sample_mean - mu)))
    
    # Checking if U is less than or equal to 1/2
    if U <= 0.5:
        count_U_le_half += 1

    # Save U values for plotting
    simulation_results.append(U)

# Calculating the simulated probability
simulated_prob = count_U_le_half / num_simulations

# Calculating log of the simulated probability
log_simulated_prob = np.log(simulated_prob)

print("Simulated Probability:", simulated_prob)
print("Log of Simulated Probability:", log_simulated_prob)

# Plotting the histogram of simulated U values
plt.hist(simulation_results, bins=30, density=True, alpha=0.6, color='blue', edgecolor='black')
plt.title('Histogram of Simulated U values')
plt.xlabel('U values')
plt.ylabel('Density')
plt.savefig("./figs/sim.png",bbox_inches='tight')
