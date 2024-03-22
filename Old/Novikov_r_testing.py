import numpy as np
from matplotlib import pyplot as plt

#Write function to calculater r from Rmax and M

def r(Rmax,M):
    print(Rmax,M)
    r = (Rmax / (2 * M) - 1)**0.5
    return(r)

#Calculate r for varying Rmax and M

Mnum = 5
Mfac = 0.01
Rnum = 1000

Rlist = np.arange(Rnum) + 1
rlist = np.zeros([Mnum,Rnum])
for i in range(Mnum):
    for j in range(Rnum):
        rlist[i,j] = r(Rlist[j], ((i + 1) * Mfac))
print(rlist)

#Plot

legend = []
for i in range(Mnum):
    legend.append("M = " + str((i+1) * Mfac))

for i in range(Mnum):
    plt.plot(Rlist, rlist[i])
plt.xlabel("Rmax")
plt.ylabel("r")
plt.legend(legend)
plt.show()