import numpy as np
import matplotlib.pyplot as plt

arr = np.array([[4], [2], [1]])
size = steps = 100

#generate function is modified and generalized compared to part a) and b)
def generate(rule_set):
    # computing the binary representation of the rule
    binary_rule = np.array([int(_) for _ in np.binary_repr(rule_set, 8)], dtype = np.int8)
    
    x = np.zeros((steps, size), dtype = np.int8)
    # generating random initial conditions
    x[0, :] = np.random.rand(size) < 0.5
    
    for i in range(steps - 1):
        x[i + 1, :] = next_gen(x[i, :], binary_rule)
    
    return x

# function which computes the next step of an elementary CA
def next_gen(x, rule):
    """ vstack concatenates the 2 arrays
    Periodic boundary conditions using np.roll - Elements that roll 
    beyond the last position are re-introduced at the first. """
    a = np.vstack((np.roll(x, 1), x, np.roll(x, -1))).astype(np.int8)
    
    # generates a pattern of left neightbor, cell and right neighbor from 0 to 7
    sum_ = np.sum(a * arr, axis = 0).astype(np.int8)
    
    return rule[7 - sum_]

# plotting
fig, axes = plt.subplots(2, 3, figsize = (16, 10))
rules = [18, 73, 90, 150, 136, 3]

for ax, rule in zip(axes.flat, rules):
    x = generate(rule)
    ax.imshow(x, interpolation='none', cmap=plt.cm.binary)
    ax.set_title("Rule {}".format(rule))
plt.savefig('evolution-rules.png')