import numpy as np
import matplotlib.pyplot as plt

n = 1  # number of neighbors
t = 100  # time limit
size = 2*t + 1
rule_num = 30
# computing the binary representation of the rule
binary_rule = format(rule_num, '0' + str(np.power(2, (2*n + 1))) + 'b')
rules = list(binary_rule)  # making a list of binary rule sets

# function which applies a certain rule set
def generate(array):
    x = ' '
    for i in array:
        x += str(i)
    j = np.power(2, (2 * n + 1)) - 1 - int(x, 2)
    return int(rules[j])

# updates the next generation
def next_gen(array):
    array_next = []
    
    for i in range(size):
        a = []
        for i in range(i - n, i + n + 1):
            a.append(array[i % size])
        array_next.append(generate(a))
        
    return array_next

arr = np.zeros(size, dtype = np.int8)
arr[int((size - 1) / 2)] = 1
a = np.array([arr])
for i in range(t):
    arr = next_gen(arr)
    # vstack concatenates 2 arrays
    a = np.vstack((a, arr))

# plotting
fig = plt.figure(dpi = 150, figsize = [10, 5])
ax = fig.add_subplot(111)
ax.set_title("Rule {}, {}".format(rule_num, binary_rule))
ax.tick_params(labelcolor = 'white', color = 'white')
img = ax.imshow(a, interpolation="nearest", cmap = plt.cm.binary)    
plt.savefig('Rule30.png')