import os

# dir_name = 'GAN_results'
dir_name = 'backup'

loss_for_graph_append = []
if os.path.exists(os.path.join(dir_name, 'GAN_loss_for_graph.txt')):
    with open(os.path.join(dir_name, 'GAN_loss_for_graph.txt'), 'r') as f:
        for line in f:
            loss_for_graph_append.append([float(line.split(' ')[0]), float(line.split(' ')[1])])


# Plot loss for graph
import matplotlib.pyplot as plt
plt.figure()
plt.title("GAN loss for graph")
plt.xlabel("Step")
plt.ylabel("Loss")
plt.plot(loss_for_graph_append)
plt.show()