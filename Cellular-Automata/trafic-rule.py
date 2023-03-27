import numpy as np
import matplotlib.pyplot as plt

arr = np.array([[4], [2], [1]])
size = steps = 100
t = 100  # time limit
rule_num = 184

# function to compute the given rule set
def generate(rule_set):
    # computing the binary representation of the rule
    binary_rule = np.array([int(_) for _ in np.binary_repr(rule_set, 8)], dtype = np.int8)
    
    x = np.zeros((steps, size), dtype = np.int8)
    x[0, :] = np.random.rand(size) < 0.5
    
    for i in range(steps - 1):
        x[i + 1, :] = next_gen(x[i, :], binary_rule)
    
    return x

# function which computes the next step of an elementary CA
def next_gen(x, rule):
    """ vstack concatenates the 2 arrays.
    Periodic boundary conditions using np.roll - Elements that roll 
    beyond the last position are re-introduced at the first. """
    a = np.vstack((np.roll(x, 1), x, np.roll(x, -1))).astype(np.int8)
    
    # generates a pattern of left neightbor, cell and right neighbor from 0 to 7
    sum_ = np.sum(a * arr, axis = 0).astype(np.int8)
    
    return rule[7 - sum_]

x = generate(rule_num)

# plotting
fig = plt.figure(dpi = 150, figsize = [16, 10])
ax = fig.add_subplot(111)
ax.tick_params(labelcolor = 'white', color = 'white')
ax.imshow(x, interpolation='nearest', cmap=plt.cm.binary)
ax.set_title("Rule {}".format(rule_num))
plt.savefig('Rule184.png')