# Guillaume Payeur (260929164)
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit as fit
plt.rcParams.update({"text.usetex": True})
plt.rcParams.update({'text.latex.preamble':[r'\usepackage{physics}']})

# Setting up constants
# n is the number of simulations for the x_rms plot
n = 2500
# t_max is the max number of steps
t_max = 10000

# Function that returns a random step that's either big or small
def random(n,t):
    if (t+1)%8 == 0:
        return (np.random.randint(0,2,size=n)*6)-3
    return (np.random.randint(0,2,size=n)*2)-1

################################################################################
# Code to create a single t vs x(t) plot
################################################################################
def single_sim(t_max):
    x = 0
    x_array = [0]
    for t in range(t_max):
        x += random(1,t)[0]
        x_array.append(x)
    return x_array

plt.style.use('seaborn-whitegrid')
plt.plot(np.arange(0,t_max+1),single_sim(t_max))
plt.xlabel('$t$')
plt.ylabel('$x(t)$')
plt.title('$x \\text{ versus } t$')
plt.show()
# ################################################################################
# # Code to create a t vs x_rms(t) plot
# ################################################################################
# Function that does n simulations with t_max steps
def sim(n,t_max):
    result = np.zeros((n))
    for t in range(t_max+1):
        result += random(n,t)
    return np.sqrt(np.mean(result**2))

# Doing n simulations from 1 step to t_max steps
x_rms = np.zeros(t_max+1)
for t in range(t_max):
    x_rms[t+1] = sim(n,t)

plt.plot(np.arange(0,t_max+1),x_rms)
plt.xlabel('$t$')
plt.ylabel('$x_{\\text{rms}}(t)$')
plt.title('$x_{\\text{rms}} \\text{ versus } t$')
plt.show()

################################################################################
# Code to do curve fitting on x_rms
################################################################################
# Defining the Angela and Donald functions
def Angela(t,eps):
    return np.sqrt(2*(1/2+eps)*t)
def Donald(t,delta):
    return t**(1/2+delta)

# Fitting both models to x_rms
best_eps = fit(Angela,np.arange(0,t_max+1),x_rms,p0=0)[0][0]
best_delta = fit(Donald,np.arange(0,t_max+1),x_rms,p0=0)[0][0]
print(best_eps)
print(best_delta)

# Calculating the mean squared error for both models
MSE_Angela = np.mean(np.square((x_rms - Angela(np.arange(0,t_max+1),best_eps))))
MSE_Donald = np.mean(np.square((x_rms - Donald(np.arange(0,t_max+1),best_delta))))

# Plotting the curve fits with the x_rms array
plt.plot(np.arange(0,t_max+1),x_rms,color='black',label='Simulation')
plt.plot(np.arange(0,t_max+1),Angela(np.arange(0,t_max+1),best_eps),color='#1AE700',label='Angela\'s model, MSE={}'.format(MSE_Angela))
plt.plot(np.arange(0,t_max+1),Donald(np.arange(0,t_max+1),best_delta),color='#FF431B',label='Donald\'s model, MSE={}'.format(MSE_Donald))
plt.xlabel('$t$')
plt.ylabel('$x_{\\text{rms}}(t)$')
plt.title('$x_{\\text{rms}} \\text{ versus } t$, compared to best fit curves')
plt.legend(frameon=True)
plt.show()
