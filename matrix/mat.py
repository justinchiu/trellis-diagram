

import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
from matplotlib.path import Path
from matplotlib.patches import PathPatch

from numpy.random import randn

def p(x, mask):
    # why is the resolution so bad?
    #plt.imsave("mat.png", x, format="png", cmap=cm.hot)
    fig,ax = plt.subplots(figsize = (20,20))
    color = cm.Purples(plt.Normalize()(x))
    grey = color.copy()
    #grey[:,:,0] = 0.6
    #grey[:,:,1] = 0.6
    #grey[:,:,2] = 0.65
    #grey[:,:,0] = 0.9
    #grey[:,:,1] = 0.9
    #grey[:,:,2] = 0.95
    #mask = mask[:,:,np.newaxis]
    #im = ax.imshow(mask * color + (1-mask) * grey)
    im = ax.imshow(x, cmap=cm.Purples)

    plt.Axes(fig, [0,0,1,1])
    ax.patch.set_edgecolor("black")
    ax.patch.set_linewidth("1")
    ax.axis("off")
    #plt.show()
    #import pdb; pdb.set_trace()
    plt.savefig("mat.png", bbox_inches="tight", pad_inches=0)

Z, X = 16, 32
M = 4
k = Z // M

x = 3 * np.abs(randn(Z, X)) + 3

# produce block structure
lengths = [4, 16, 8, 4]
mask = np.zeros((Z, X))
for m in range(M):
    mask[m*k:(m+1)*k,sum(lengths[:m]):sum(lengths[:m+1])] = 1

p(x*mask, mask)
