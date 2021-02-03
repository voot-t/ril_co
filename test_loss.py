
import matplotlib 
import matplotlib.pyplot as plt
import pylab
import torch 
import torch.nn as nn
import torch.nn.functional as F
import numpy as np 

# matplotlib.rcParams['pdf.fonttype'] = 42
# matplotlib.rcParams['ps.fonttype'] = 42

from a2c_ppo_acktr.algo.ail_utils import *

def test_all():
    logistic_loss = Logistic_Loss()
    hinge_loss = Hinge_Loss() 
    unhinged_loss = Unhinged_Loss() 
    sigmoid_loss = Sigmoid_Loss() 
    nlogistic_loss = Normalized_Logistic_Loss()
    nhinge_loss = Normalized_Hinge_Loss() 
        
    xy_lim = 3
    linewidth = 3
    fontsize = 15
    f = plt.figure(figsize=(6, 5)) 
    ax = f.gca() 
    z = torch.from_numpy(np.linspace(-xy_lim, xy_lim, 1000))

    logistic = logistic_loss(z, False)
    hinge = hinge_loss(z, False)
    unhinged = unhinged_loss(z, False) 
    sigmoid = sigmoid_loss(z, False) 
    nlogistic = nlogistic_loss(z, False) 
    nunhinge = nhinge_loss(z, False) 
            
    ax.plot(z, z * 0, color="k", linewidth=1)
    ax.plot(z*0, z, color="k", linewidth=1)

    ax.plot(z, logistic, color="r", label="Logistic", linestyle="-", linewidth=linewidth)
    ax.plot(z, hinge, color="lime", label="Hinge", linestyle="--", linewidth=linewidth)
    ax.plot(z, sigmoid, color="violet", label="Sigmoid", linestyle="-.", linewidth=linewidth)
    ax.plot(z, unhinged, color="b", label="Unhinged", linestyle=":", linewidth=linewidth) 
    ax.plot(z, nlogistic, color="goldenrod", label="Normalized logistic", linestyle="-", linewidth=linewidth)
    ax.plot(z, nunhinge, color="deepskyblue", label="Normalized hinge", linestyle="--", linewidth=linewidth)

    ax.legend(prop={"size":fontsize}, frameon=True, framealpha=1, loc = 'lower left')  
    # ax.grid(True, which='major')
    # ax.xaxis.grid(False, which='minor')

    plt.xlabel(r"$z$", fontsize=fontsize+1)      
    plt.ylabel(r"$\ell(z)$", fontsize=fontsize+1)
    
    plt.xlim(-xy_lim,xy_lim)
    plt.ylim(-xy_lim,xy_lim)
    plt.tight_layout()

    f.savefig("./figures/loss.pdf", bbox_inches='tight', pad_inches = 0)

    plt.show()

if __name__ == "__main__":
    test_all()