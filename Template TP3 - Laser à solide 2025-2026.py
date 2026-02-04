
"""
TP 3 : Modélisation d'un laser à solide en régime continu et impulsionnel

"""

from numpy import *
import matplotlib.pyplot as plt
import time

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

   
# Fonctions de conversion Flux <-> Puissance

def Flux(P, wl): return P*wl/(pi*r*r*h*c)

def Puissance(F, wl): return F*(pi*r*r*h*c)/wl


# Calcul de la population N2 en régime stationnaire

N2 = 

print("Valeur de N2 au seuil en régime stationnaire = ", N2, "cm-3")

# Calcul du flux et de la puissance de pompage au seuil en régime stationnaire

Fpseuil = 

Pseuil940 = 

print("Puissance de pompage au seuil =", Pseuil940, "W")


# Calcul de la puissance de sortie du laser en régime stationnaire en fonction de la puissance de pompage

def Plaser(P):

# Définition f(x,y) = f1(y1, y2) pour le calcul de N2 avec y1 = N2 et y2 = Flux1030

def f1(y1, y2, F940) :

# Définition f(x,y) = f2(y1, y2) pour le calcul de Flux1030 avec y1 = N2 et y2 = Flux1030

def f2(y1,y2) :

# Evolution de la population N2 en régime transitoire

def EulerN2(P940, dt, tmax):

# Evolution de la puissance de sortie du laser en régime transitoire

def EulerLaser(P940, dt, tmax):


