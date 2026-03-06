import numpy as np
from matplotlib import pyplot as plt





#Constants
alpha = 0.5 #cm-1
L = 6 #cm



#Questions

#1
"""
If a Z is a random variable following an exponential pdf

Z(z) = lambda*exp(-lambda*z)

The CDF (Fonction de Repartition) is then given by 

FZ(z) = integral_{-inf}^{z}z(u)du = lambda/(-lambda)[exp(-lambda*u)]_{-inf}^{z}
FZ(z) = (-1)*[exp(-lambda*z) - 1] = 1 - [exp(-lambda*z)

The sampling rate lambda can be taken as the linear coefficient loss alpha

Another way to see it

The transmited power to a position M is homogenous to the power that is transmitted to that position
over the total incident power

P(Z>M) = Transmitted/Incident = exp(-alpha*M) , then the pdf is found by derivating this

"""
#2
"""
FZ(z) = 1 - [exp(-lambda*z) -> ln(1 - FZ) = -lambda*z -> z = -ln(1 - FZ)/lambda = FZ^{-1}

By the same sampling law that we saw earlier in the Monte Carlo TP, tp2, we know that if a random variable Z follows
a certain pdf , then a sample of Z can be found as : 

Z_i = FZ^{-1}(U_i)

Where U_i is a sample from a uniform law with 0,1 hi lo

"""
#4
"""
We do the integral of the solid angle from theta from 0 to pi/2

Delta Omega = 2pi * integral_{0}^{theta}sin(t)dt = 2pi * [-cos(t)]_{0}^{theta} = 2pi * [1 - cos(theta)]

Donc

FT = (1/2) * [1 - cos(theta)]

cos(theta) = 1 - 2FT = FT^(-1)

"""
class Photon :
    def __init__(self , z):
        self.trajectory = []
        self.z = z
        self.theta = None
        self.phi = None

def ZCDF(u):
    return 1 - np.exp(-alpha*u)

def inverse_ZCDF(u):
    return (-np.log(1 - ZCDF(u) ))/alpha

def thetaCDF(u):
    return (1/2) * [1 - np.cos(u)]

def inverse_costhetaCDF(u):
    return 1 - 2*thetaCDF(u)



def diffusion(n_photons):
    z_uniform_samples = np.random.uniform(low = 0 , high=1 , size = n_photons)

    print(z_uniform_samples)

    z_samples = inverse_ZCDF(z_uniform_samples)

    print(z_samples)

diffusion(10)
