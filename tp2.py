import  numpy as np
from matplotlib import pyplot as plt
import tp1 


class Exercice():

    def __init__(self , low=0 , high=1 ):
        self.low = low
        self.high = high
        self.densite_probabilite = None
        
 
    def g(self , x):
        return 2 * (np.exp(x) - 1)/(np.e - 1)
        

    def h(self , x):
        return self.g(x) / self.densite_probabilite()
    
    def integrand(self , x):
        return self.h(x)**2 * self.densite_probabilite()
    
    def G(self , x):
        return 2/(np.e - 1) * (np.exp(x) - x)

    def densite_uniforme(self):
        return 1/(self.high-self.low)
    
    def set_densite_probabilite(self , p) :
        self.densite_probabilite = p


if __name__ == "__main__":
    e = Exercice(low = 0 , high= 1 )
    e.set_densite_probabilite(e.densite_uniforme)
    exact_value = e.G(e.high) - e.G(e.low)
    integrator = tp1.Exercice()
    trapeze , _ = integrator.ex_integration_numerique(e.integrand , e.low , e.high , N = 100 , M = 'trapeze')
    variance = trapeze - exact_value ** 2
    print(variance)
    # Import integrate function from tp1 and integrate h**2 * f
    # std = trapeze - I**2
    