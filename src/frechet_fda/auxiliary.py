
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import truncnorm

class afunc:

    def __init__(self, x: np.ndarray, y: np.ndarray):
        self.x = x
        self.y = y

    def integrate(self):
        x = np.copy(self.x)
        y = np.copy(self.y)

        int_x = x
        point_distance = np.diff(x)
        int_y = np.zeros_like(y)
        int_y[1:] = np.cumsum(y[:-1]*point_distance)
        return afunc(int_x, int_y)
    
    def integrate_sequential(self):
        x = np.copy(self.x)
        y = np.copy(self.y)

        int_x = x
        point_distance = np.diff(x)
        int_y = np.zeros_like(y)
        for i in range(1,x.shape[0]):
            int_y[i] = int_y[i-1] + y[i-1]*point_distance[i-1]
        return afunc(int_x, int_y)

    def derive(self):
        x = np.copy(self.x)
        y = np.copy(self.y)

        d_x = x
        d_y = np.zeros_like(y)
        d_y[:-1]=np.diff(y)/np.diff(x)
        d_y[-1] = d_y[-2]
        return afunc(d_x, d_y)

    def invert(self):
        inv_x = np.copy(self.y)
        inv_y = np.copy(self.x)
        return afunc(inv_x,inv_y)
    
    def log(self):
        log_x = np.copy(self.x)
        log_y = np.log(np.copy(self.y))
        return afunc(log_x,log_y)
    
    def exp(self):
        exp_x = np.copy(self.x)
        exp_y = np.exp(np.copy(self.y))
        return afunc(exp_x,exp_y)
    
    def compose(self, other):
        comp_x = np.copy(other.x)
        comp_y = np.interp(other.y,self.x,self.y)
        return afunc(comp_x,comp_y)

    def plot(self, restricted = 0):
        if restricted > 0:
            rest_x = self.x[restricted:-restricted]
            rest_y = self.y[restricted:-restricted]
        else:
            rest_x = self.x
            rest_y = self.y
        plt.plot(rest_x,rest_y)
        plt.grid()

    def compare(self, other):
        fig, ax = plt.subplots()
        ax.plot(self.x,self.y,color="red",label="self")
        ax.plot(other.x,other.y,color="blue",label="other")
        plt.legend()
        plt.grid()
        plt.show()

    def __sub__(self, val: float | int):
        x = np.copy(self.x)
        y = np.copy(self.y)
        if isinstance(val, float | int):
            return afunc(x,y-val)
        elif isinstance(val, afunc):
            return afunc(x,y-val.y)

    def __add__(self, val: float | int):
        x = np.copy(self.x)
        y = np.copy(self.y)
        if isinstance(val, float | int):
            return afunc(x,y+val)
        elif isinstance(val, afunc):
            return afunc(x,y+val.y)
        
def make_pdf(left: float, right: float, mu: float = 0, sigma: float = 1, grid_size: int = 1000) -> afunc:
    left_std = (left - mu) / sigma
    right_std = (right - mu) / sigma
    pdf_x = np.linspace(left,right,grid_size)
    point_distance = (right-left)/(grid_size-1)
    pdf_y = truncnorm.pdf(
        x = pdf_x,
        a = left_std,
        b = right_std,
        loc = mu,
        scale = sigma
    )
    return afunc(pdf_x,pdf_y)