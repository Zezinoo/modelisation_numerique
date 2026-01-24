import  numpy as np
from matplotlib import pyplot as plt


class Exercice():

    def __init__(self , a , b , n):
        self.a = a
        self.b = b
        self.n = n
        random_array = np.random.uniform(low = a, high=b , size = n)
        

    @staticmethod 
    def g(x):
        return 2 * (np.exp(x) - 1)/(np.e - 1)
        

    def h(self , x):
        return self.g(x) / self.densite_uniforme()

    @staticmethod
    def G(x):
        return 2/(np.e - 1) * (np.exp(x) - x)

    def densite_uniforme(self):
        return 1/(self.b-self.a)


if __name__ == "__main__":
    e = Exercice()
    exact_value = e.G(1) - e.G(0)
    print(exact_value)
    # Import integrate function from tp1 and integrate h**2 * f
    # std = trapeze - I**2
    