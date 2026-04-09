import numpy as np
from matplotlib import pyplot as plt

y0 = 0.3
K = 8
R = 0.02 #s-1


def f_euler(R,K,y):
    return R*y - R*K*y**2

def f_exact(t):
    return (y0*np.exp(R*t))/(1 + K*y0*(np.exp(R*t)-1))

def f_exact_modifie(t):
    if t >= 50:
        constant = 30
    else:
        constant = K
    return (y0*np.exp(R*t))/(1 + constant*y0*(np.exp(R*t)-1))



def Euler_pop(N,K,R,y0,tmax):
    dt = tmax/N
    pops = []
    yn = y0
    pops.append(y0)
    counter = 0
    while counter < N:
        yn1 = yn + dt*f_euler(R,K,yn)
        pops.append(yn1)
        yn = yn1
        counter = counter + 1
    
    return pops


def Euler_pop_modifie(N,K,R,y0,tmax):
    dt = tmax/N
    pops = []
    pops.append(y0)
    yn = y0
    counter = 0
    while counter < N:
        if dt*counter >= 50:
            constant = 30
        else :
            constant = K
        yn1 = yn + dt*f_euler(R,constant,yn)
        pops.append(yn1)
        yn = yn1
        counter = counter + 1
    
    return pops


tmax = 100
N = 100
# Partie 1

times = np.linspace(0,tmax,N+1)

pops = Euler_pop(N,K,R,y0,tmax)

plt.plot(times , f_exact(times) , label = "exact")
plt.plot(times , pops , label = "euler")
plt.xlabel("Temp (s)")
plt.ylabel("Population")
plt.title("Equation de population")

plt.legend()
plt.show()

# Partie 2

pops_mod = Euler_pop_modifie(N,K,R,y0,tmax)

exact_modifie = []
for t in times:
    exact_modifie.append(f_exact_modifie(t))

plt.plot(times , exact_modifie , label = "exact")
plt.plot(times , pops_mod , label = "euler")

plt.xlabel("Temp (s)")
plt.ylabel("Population")
plt.title("Equation de population modifié")


plt.legend()
plt.show()