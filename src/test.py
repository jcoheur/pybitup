import os
import matplotlib.pyplot as plt
import pickle 
import random
import numpy as np
from scipy.stats import gaussian_kde

import pyBIT.Metropolis_Hastings_Inference as MH

"""mon_fichier=open("fichier.txt", "w")

# contenu = mon_fichier.read()
# print(contenu)

mon_fichier.write("Salut je suis un gros pd\n")
mon_fichier.write("j\'ai {} ans".format(str(50))) 


mon_fichier.close()

X = [590,540,740,130,810,300,320,230,470,620,770,250]
Y = [32,36,39,52,61,72,77,75,68,57,48,48]

plt.scatter(X,Y)
plt.show()

i = 0
X = []
while i < 10000:

	rn=random.gauss(0, 1)
	X.append(rn)
	i += 1
	
density = gaussian_kde(X)
xs = np.linspace(-5,5,100)
density.covariance_factor = lambda : 1
density._compute_covariance()
plt.plot(xs, density(xs))
plt.show()"""



os.system("pause")