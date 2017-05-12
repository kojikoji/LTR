import plotly.plotly as py
import plotly.graph_objs as go
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
#transform to [[x,y,z]...] [[x...],[y...],[z...]]
#plot
def plot_sample_list(sample_list,lim_val=10):
    sample_mat = np.array(sample_list)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(sample_mat[:,0], sample_mat[:,1], sample_mat[:,2], c='r', marker='o')
    ax.set_xlim(-lim_val, lim_val)
    ax.set_ylim(-lim_val, lim_val)
    ax.set_zlim(-lim_val, lim_val)
    plt.show()
    Axes3D.plot()
