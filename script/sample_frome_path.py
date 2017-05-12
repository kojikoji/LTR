from numpy.random import *
import numpy as np
import math
from sample_plotter import plot_sample_list
class Path:
    # set the path num
    def __init__(self,var_num,beta_list,time_scale = 5, poly_dim = 3):
        self.var_num = var_num
        self.beta_list = beta_list
        self.to_init = True
        self.time_scale = 0.3
        self.poly_dim = poly_dim
        self.freq= 1
        self.make_path()
    def time_func(self,t,rval,coff_val):
        return(math.sin(rval*math.pi*2*self.freq*(t - coff_val)))
    # initial lize function and constitute a path
    def make_path(self):
        self.r_val = 2*(rand(self.var_num) - 0.5)
        self.coff_val = 2*(rand(self.var_num) - 0.5)
        self.to_init = False
    # one sample from initialized path
    def sample_from_path(self):
        t = rand()
        mu = [self.time_func(t,self.r_val[i],self.coff_val[i]) for i in range(self.var_num)]
        sample =  multivariate_normal(mu, np.diag(self.beta_list))
        return(np.array(sample))
# set of paths witin which variables are synchronized
class Mixed_Path():
    def __init__(self,var_num_list,beta_mat):
        # number of variable for each path
        self.var_num_list = var_num_list
        #beta for 
        self.beta_mat = beta_mat
        self.path_list = []
        self.make_paths()
    # register each path
    def make_paths(self):
        for (i,var_num) in enumerate(self.var_num_list):
            print(var_num)
            self.path_list.append(Path(var_num,self.beta_mat[i]))
    def sample_from_paths(self):
        sample = []
        for path in self.path_list:
            sub_sample = path.sample_from_path()
            sample.extend(sub_sample)
        return(sample)
def test_path():
    dims = 3
    path = Path(dims,np.array([0.2,0.2,0.2])*1.0e-1)
    dim_array = [[] for i in range(dims)]
    print(dim_array)
    sample_list = []
    for i in range(100):
        sample = path.sample_from_path()
        sample_list.append(sample)
    plot_sample_list(sample_list,lim_val=2)
def test_paths():
    dim_list = [1,2]
    val = 0.02
    beta_mat = [[val],[val,val]]
    mixed_path = Mixed_Path(dim_list,beta_mat)
    sample_list = []
    for i in range(100):
        sample = mixed_path.sample_from_paths()
        sample_list.append(sample)
    print(sample_list[0])
    plot_sample_list(sample_list,lim_val=2)    
if __name__=="__main__":
    test_paths()
