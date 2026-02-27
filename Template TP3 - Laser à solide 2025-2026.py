
"""
TP 3 : Modélisation d'un laser à solide en régime continu et impulsionnel

"""

import matplotlib.pyplot as plt
import time
import numpy as np

# Debut du decompte du temps
start_time = time.time()

# Données numériques
lambda940 = 940.0E-9                                # Longueur d’onde de pompage en m
lambda1030 = 1030.0E-9                              # Longueur d’onde du laser en m
Nt = 1.0E20                                         # Concentration en ions Yb3+ en cm-3	       
tau = 1.0E-3                                        # Durée de vie du niveau émetteur en s		
r = 0.01                                            # Rayon des faisceaux cylindriques en cm 		
L = 1.0                                             # Longueur de la cavité résonante en cm 		
alpha = 0.02                                        # Coefficient de perte dans la cavité en cm-1			
R1 = 1.0                                            # Coefficient de réflexion du miroir d'entrée M1 à 1030nm
R2 = 0.98                                           # Coefficient de réflexion du miroir de sortie M2 à 1030nm
sea940 = 2.0E-20                                    # Section efficace d’absorption à 940nm en cm2			
sea1030 = 0.1E-20                                   # Section efficace d’absorption à 1030nm en cm2 			
see1030 = 0.3E-20                                   # Section efficace d’émission à 1030nm en cm2 			
c = 3.0E8                                          # Vitesse de la lumière dans le vide en m.s-1 
n = 1.5                                             # Indice de réfraction du milieu amplificateur
h = 6.63E-34                                        # constante de Planck en J/s

fes = 0.01                                          # Fraction d'émission spontanée pour initier le flux de photons à 1030nm

QS = 1                                              # Transmission du modulateur de perte intra-cavité : QS = 0 ou 1

gamma = alpha - np.log(R1*R2)/(2*L)                    # Coeff de perte total de la cavité en cm-1



# Calcul de la puissance de sortie du laser en régime stationnaire en fonction de la puissance de pompage

def Plaser(P):
    flux = Flux(P , 940e-9)
    laser_flux = 1/2 * (flux*sea940*N1 - N2/tau)/(see1030*N2 - sea1030*N1)
    return Puissance(laser_flux , 1030e-9)

# Définition f(x,y) = f1(y1, y2) pour le calcul de N2 avec y1 = N2 et y2 = Flux1030

def f1(y1, y2, F940) :
    #DN2/dt
    return F940*sea940*(Nt-y1) - 2*y2*(see1030*y1 - sea1030*(Nt - y1)) -y1/tau

# Définition f(x,y) = f2(y1, y2) pour le calcul de Flux1030 avec y1 = N2 et y2 = Flux1030

def f2(y1,y2,F940,beta = 1e-2) :
    #Dphi1030/dt
    eq = y2*(see1030*y1 - sea1030*(Nt - y1))*c*100/n - gamma*y2*c*100/n + beta*y1/tau
    #Multiplying by 100 to turn lightspeed units into cm to be in accordance with other values
    return eq

# Evolution de la population N2 en régime transitoire

def EulerN2(P940, dt, tmax):
    t = 0
    evolution_N2 = []
    yn = 0 #N2 population starts at 0
    while t < tmax:
        if t < 3e-3:
            flux940 = Flux(P940 , 940e-9)
        else :
            flux940 = 0

        yn1 = yn + dt*f1(yn ,0 ,flux940)
        yn = yn1
        t = t + dt
        evolution_N2.append((t , yn))
    return evolution_N2

def exact_solution(p940 , t):
    #Solve the ODE dn2/dt = k*(nt-n2) - 1/tau
    k = Flux(p940 , 940e-9) * sea940
    return (k*Nt*tau)/(k*tau + 1) * (1 - np.exp(-(k + 1/tau)*t))

# Evolution de la puissance de sortie du laser en régime transitoire


def EulerLaser(P940, dt, tmax):
    flux940 = Flux(P940 , 940e-9)
    t = 0
    evolution_flux = []
    yn = 0 # LaserFlux starts at 1030
    xn = 0 # N2 starts at 0
    while t < tmax:
        xn1 = xn + dt*f1(xn , yn , flux940)
        yn1 = yn + dt*f2(xn1 , yn , flux940)
        xn = xn1
        yn = yn1
        t = t + dt
        evolution_flux.append((t,yn))
    return evolution_flux


def RungeKutta(P940 , dt , tmax):
    flux940 = Flux(P940 , 940e-9)
    t = 0
    evolution_flux = []
    yn = 0 # LaserFlux starts at 1030
    xn = 0 # N2 starts at 0
    while t < tmax:
        #Solving coupled population equation
        xn1 = xn + RungeKuttaSteps(f1 , xn , yn , flux940 , dt)
        yn1 = yn + RungeKuttaSteps(f2 , xn1 , yn , flux940 , dt)
        xn = xn1
        yn = yn1
        t = t + dt
        evolution_flux.append((t,yn))
    return evolution_flux
        

def RungeKuttaSteps(f , x , y , flux940 , dt):
    k1 = dt*f(x,y,flux940)
    k2 = dt*f(x + dt/2 , y + k1/2,flux940)
    k3 = dt*f(x + dt/2 , y + k2/2,flux940)
    k4 = dt*f(x + dt , y + k3,flux940)

    methodstep = 1/6 * (k1 + 2*k2 + 2*k3 + k4)
    return methodstep


def RungeKuttaQSwitch(P940 , dt , tmax):
    flux940 = Flux(P940 , 940e-9)
    t = 0
    evolution_flux = []
    yn = 0 # LaserFlux starts at 1030
    xn = 0 # N2 starts at 0
    while t < tmax:
        #Solving coupled population equation
        xn1 = xn + RungeKuttaSteps(f1 , xn , yn , flux940 , dt)
        if t < 40e-6:
            yn1 = 0
        else:
            yn1 = yn + RungeKuttaSteps(f2 , xn1 , yn , flux940 , dt)
        xn = xn1
        yn = yn1
        t = t + dt
        evolution_flux.append((t,yn))
    return evolution_flux



# Fonctions de conversion Flux <-> Puissance

def Flux(P, wl): return P*wl/(np.pi*r*r*h*c)

def Puissance(F, wl): return F*(np.pi*r*r*h*c)/wl

def calculateFWHM(array):
    arr = np.array(array)
    maximum = np.max(arr)
    half_max = maximum/2
    idx_max = np.argmax(arr)

    left_fwhm_idx = np.argmin(np.abs(arr - half_max)[:idx_max])
    right_fwhm_idx = np.argmin(np.abs(arr - half_max)[idx_max:]) + idx_max


    return left_fwhm_idx , right_fwhm_idx , half_max



if __name__ == "__main__" :
    # Calcul de la population N2 en régime stationnaire
    """
    dphi/dt = phi(sigmae*N2 - sigmaa*N1)*c/n - gamma * phi * c/n
    En regime stationaire dphi/dt = 0
    N1 = Nt - N2

    N2*(sigmae + sigmaa) = sigmaa Nt + gamma
    N2 = (sigma_a*Nt + gamma)/(sigma_e + sigma_a) 
    """


    N2 = (gamma + sea1030* Nt )/(sea1030 + see1030)
    N1 = Nt - N2

    print("Valeur de N2 au seuil en régime stationnaire = ", N2, "cm-3")

    # Calcul du flux et de la puissance de pompage au seuil en régime stationnaire
    # Dans le seuil ; le flux a 1030 nm est zero

    Fpseuil = N2/(tau*sea940*(Nt - N2))
    Pseuil940 = Puissance(Fpseuil , 940e-9)


    print("Puissance de pompage au seuil =", Pseuil940, "W")

    #Partie B

    pumping = np.linspace(0,10,1000)
    output_powers = [max(Plaser(p)*(1-R2),0) for p in pumping]

    plt.plot(pumping , output_powers )
    plt.title("Laser Power x Pumping Power")
    plt.show()

    results = EulerN2(P940=5 ,dt = 10e-6 , tmax=10e-3)
    
    times = [r[0] for r in results]
    population = [r[1] for r in results]
    exact = [exact_solution(5,t) for t in times if t <= 3e-3]
    times2 = [t for t in times if t <= 3e-3]

    plt.plot(times , population , label = "N2(t)")
    plt.plot(times2,exact , label = "exact solution")
    plt.title("N2(t)")
    plt.show()  

    #Partie C
    
    results_flux = EulerLaser(P940=50,dt = 100e-9 , tmax = 100e-6) # Divergence with slow steps
    times = [r[0] for r in results_flux]
    flux = [r[1] for r in results_flux]
    print(flux[-1])
    stationary_flux = Flux(Plaser(50), 1030e-9)
    #plt.plot(times,flux , label = "euler")
    plt.plot(times , np.ones_like(times)*stationary_flux)
    

    runge_kutta_results = RungeKutta(P940 = 50 , dt = 100e-12,tmax=100e-6)
    times = [r[0] for r in runge_kutta_results]
    flux = [r[1] for r in runge_kutta_results]
    print(flux[-1])
    stationary_flux = Flux(Plaser(50), 1030e-9)
    plt.plot(times,flux , label = "runge-kutta")
    #plt.plot(times , np.ones_like(times)*stationary_flux)
    plt.legend()
    plt.show()
    
    #Runge Kutta accepts slower steps to the order of 100e-11

    #Partie D
    runge_kutta_results = RungeKuttaQSwitch(P940 = 50 , dt = 1e-10,tmax=100e-6)
    times = [r[0] for r in runge_kutta_results if (r[0] >= 40e-6 and r[0] <= 40e-6 + 100e-9 )]
    flux = [r[1] for r in runge_kutta_results if (r[0] >= 40e-6 and r[0] <= 40e-6 + 100e-9 )]
    print(flux[-1])
    stationary_flux = Flux(Plaser(50), 1030e-9)
    plt.plot(times,flux , label = "runge-kutta")
    plt.plot(times , np.ones_like(times)*stationary_flux)
    l , r , half_max = calculateFWHM(flux)
    fwhm = times[r] - times[l]
    plt.hlines(half_max , times[l] , times[r] , color = "red" , label=f"FWHM = {fwhm:.2e}")
    plt.legend()
    plt.show()

    