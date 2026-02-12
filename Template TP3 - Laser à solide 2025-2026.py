
"""
TP 3 : Modélisation d'un laser à solide en régime continu et impulsionnel

"""

from numpy import *
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

gamma = alpha - log(R1*R2)/(2*L)                    # Coeff de perte total de la cavité en cm-1



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

def f2(y1,y2) :
    pass

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
        if yn < 1e10:
            pass
        t = t + dt
        evolution_N2.append((t , yn))
    return evolution_N2

# Evolution de la puissance de sortie du laser en régime transitoire

def exact_solution(p940 , t):
    #Sort out the constant values in the exponential solution
    k = Flux(p940 , 940e-9) * sea940
    exponenent = (-k*tau*t -t)/tau
    return -(np.exp(exponenent))/(k*tau+1) + (k*Nt*tau)/(k*tau + 1) 

def EulerLaser(P940, dt, tmax):
    pass


# Fonctions de conversion Flux <-> Puissance

def Flux(P, wl): return P*wl/(pi*r*r*h*c)

def Puissance(F, wl): return F*(pi*r*r*h*c)/wl





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

    pumping = np.linspace(0,10,1000)
    output_powers = [Plaser(p)*(1-R2) for p in pumping]

    plt.plot(pumping , output_powers )
    plt.show()

    results = EulerN2(P940=5 ,dt = 10e-6 , tmax=10e-3)
    
    times = [r[0] for r in results]
    population = [r[1] for r in results]
    exact = [exact_solution(5,t) for t in times if t <= 3e-3]
    times2 = [t for t in times if t <= 3e-3]

    plt.plot(times , population)
    plt.plot(times2,exact)
    plt.show()
