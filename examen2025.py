import math
import numpy as np
from matplotlib import pyplot as plt


wl = 1550e-9
Nt = 2e18
tau = 1e-3
r = 0.15
L = 5
sea = 7e-19
c = 3e8
h = 6.63e-34

def calc_pop_N1(Nt , F , sea , tau):
    # N2 = Nt - N1
    # N1(F*sea + 1/tau) =Nt/tau
    return Nt/(tau) * 1/(F*sea + 1/tau)

def Puissance(F):
    return F*c/wl*h*np.pi*r**2

def Flux(P):
    return P/((c / wl) * h * np.pi * r**2)


def f_flux(flux , N1):
    return -flux*sea*N1


def Euler(Pi , dz=0.2):
    populations = []
    fluxes = []
    counter = 0
    yn = Flux(Pi)
    xn = calc_pop_N1(Nt , Flux(Pi) , sea , tau)
    fluxes.append(yn)
    populations.append(xn)

    while counter < L:
        yn1 = yn + dz*f_flux(yn , xn )
        xn1 = calc_pop_N1(Nt , yn1 , sea , tau)
        yn = yn1
        xn = xn1
        fluxes.append(yn)
        populations.append(xn)
        counter = counter + dz

    return fluxes , populations

def Trans(Pi , dz):
    last_flux = Euler(Pi , dz=dz)[0][-1]

    if Pi == 0:
        transmission = 0
    else :

        transmission = Puissance(last_flux)/Pi

    return transmission




distances = np.arange(0 , L + 0.2 ,step=0.2)

fluxes , pops = Euler(10)

plt.plot(distances , fluxes)
plt.show()

plt.plot(distances , pops)

plt.show()

powers = list(range(0,110,10))
ts = [Trans(p , dz=0.2) for p in powers]

plt.plot(powers , ts)
plt.show()


### 2

def f(x):
    if x>=-1 and x<=1:
        val = 2 - abs(3*x**3 - 1)
    else : 
        val = 0
    return val

def integrate(x1,x2,f,subspaces=1001):
    intervals = np.linspace(x1,x2,num=subspaces)
    area = 0
    for i in range(len(intervals) - 1):
        a = intervals[i]
        b = intervals[i + 1]
        area += (b-a)/6 *(f(a) + f(b) + 4*f((a+b)/2))
    return area 

value = integrate(-2,2,f)
print(value)

xs = np.linspace(-2, 2, 10001)
ys = np.array([f(x) for x in xs])

print(np.trapezoid(ys , xs))

exact = (5-3**(2/3))/2

subspaces = 1000
diff = abs(integrate(-2,2,f, subspaces=subspaces+1) - exact)
while diff > 1e-4:
    subspaces = subspaces + 2
    numeric = integrate(-2,2,f,subspaces=subspaces+1)
    diff = abs(exact-numeric)

print(subspaces)

N = 2
while True:
    I_num = integrate(-2, 2, f, subspaces=N)
    err = abs(I_num - exact)
    if err < 1e-4:
        print("N minimo =", N)
        print("Integral =", I_num)
        print("Erro =", err)
        break
    N += 2

