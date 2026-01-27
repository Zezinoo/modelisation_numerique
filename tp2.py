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
        return self.g(x) / self.densite_probabilite(x)
    
    def integrand(self , x):
        return self.h(x)**2 * self.densite_probabilite(x)
    
    def G(self , x):
        return 2/(np.e - 1) * (np.exp(x) - x)

    def densite_uniforme(self,x):
        return 1/(self.high-self.low)
    
    def set_densite_probabilite(self , p) :
        self.densite_probabilite = p

    @staticmethod
    def calculate_optimal_n ( std ,u , error):
        return (u * std / error)**2

    def integrationMC(self , f , a , b , n , densite = None , cdf = None):
        """
        When using a general distribution function , a random sample can be found
        by
        Y is a random uniform value
        X = cdf^{-1}(Y)
        where is cdf is the cumulative distribution function of the density prob
        """


        if densite is None:
            densite = self.densite_uniforme


        def h(x):
            return f(x) / densite(x)
        
        sample = np.random.uniform(size = n , low = a , high = b)

        if cdf is not None:
            sample = cdf(sample)

        integral = h(sample).sum()/n
        integral_h_squared = (h(sample)**2).sum() / n
        variance = integral_h_squared - integral**2
        return integral , variance


def partie1(e):
    e.set_densite_probabilite(e.densite_uniforme)
    exact_value = e.G(e.high) - e.G(e.low)
    integrator = tp1.Exercice()
    trapeze , _ = integrator.ex_integration_numerique(e.integrand , e.low , e.high , N = 100 , M = 'trapeze')
    variance = trapeze - exact_value ** 2
    print("Exact")
    print(exact_value)
    print(variance)
    optimal_n = e.calculate_optimal_n(np.sqrt(variance) , u = 1.96 , error = 0.01 )
    optimal_n = int(np.ceil(optimal_n))
    print("Optimal N")
    print(optimal_n)
    m_integral , m_variance = e.integrationMC(e.g , 0 ,1 , optimal_n)
    print("Monte Carlo")
    print(m_integral)
    print(m_variance)

    iters = [i for i in range(100,100000,100)]
    m_values = list(map(lambda x  : e.integrationMC(e.g , 0 ,1 , x)[0] , iters))

    plt.plot(iters , m_values)
    plt.hlines(y = exact_value , xmin= iters[0] , xmax=iters[-1] , color = "red")
    plt.show()



    xs = np.linspace(0 , 1 , 10000)
    plt.plot(xs , e.g(xs) , label = "original")
    plt.plot(xs , e.densite_uniforme(xs)*np.ones_like(xs) , label = "uniform")
    plt.plot(xs , 2*xs , label = "2x")
    plt.legend()
    plt.show()

    new_density = lambda x : 2*x
    iters2 = [i for i in range(100,100000,100)]
    m_values2 = list(map(lambda x  : e.integrationMC(e.g , 0 ,1 , x , densite = new_density , cdf = np.sqrt)[0] , iters))


    plt.plot(iters2 , m_values2)
    plt.hlines(y = exact_value , xmin= iters[0] , xmax=iters[-1] , color = "red")
    plt.show()

    error_uniform = np.abs((m_values - exact_value) / exact_value)
    error_2x = np.abs((m_values2 - exact_value) / exact_value)

    plt.plot(iters  , error_uniform , label = "Uniform")
    plt.plot(iters  , error_2x , label = rf"$2x$")
    plt.legend()
    plt.show()

class MomentOfInertia:

    def __init__(self , a , R1 , R2, R3):
        self.a = a
        self.R1 = R1
        self.R2 = R2
        self.R3 = R3
        self.low = -R3
        self.high = R3

    def is_in_outside_circle(self,x,y):
        in_outer = lambda x , y : x**2 + y**2 <= self.R3**2
        in_inner = lambda x , y : x**2 + y**2 >= self.R2**2

        return in_outer(x,y) and in_inner(x,y)
    
    def is_in_center(self , x,y):
        not_in_inner = lambda x , y : x**2 + y**2 >= self.R1**2
        in_horizontal = lambda x , y : abs(y) <= self.a
        in_vertical = lambda x , y : abs(x) <= self.a

        return not_in_inner(x,y) and (in_horizontal(x,y) or in_vertical(x,y))
    
    def is_in_figure(self , x , y):
        return self.is_in_center( x,y) or self.is_in_outside_circle( x,y)
    
    def pho(self , x ,y ):
        return 1 if self.is_in_figure(x,y) else 0
    
    def g(self , x,y):
        return(x**2 + y**2)*self.pho(x,y)
    
    def integrationMC(self , f, n , n_pixels , density = None , inverse_cdf = None):
        sample = np.random.uniform(low = self.low , high = self.high , size = (n,2))
        if inverse_cdf is not None:
            sample =  inverse_cdf(sample)
        if density is None:
            density = lambda x , y : 1/(self.high - self.low)**2
        
        def h(x,y):
            return f(x,y) / density(x,y)
        
        integral = np.array([h(x,y) for x,y in sample]).sum()/n

        integral_h_squared = np.array([h(x,y)**2 for x,y in sample]).sum()/n
        variance = integral_h_squared - integral**2

        grid = self.create_image_grid( n_pixels)

        grid = self.create_image_grid(n_pixels)

        for x, y in sample:
            x_p, y_p = self.to_pixel_coords((x, y), n_pixels)
            grid[y_p, x_p] = 0

        plt.imshow(grid, cmap='gray', origin='lower')
        plt.show()

    def create_image_grid(self, n_pixels):
        grid = np.zeros((n_pixels, n_pixels), dtype=float)

        x_coords = np.linspace(-self.R3, self.R3, n_pixels)
        y_coords = np.linspace(-self.R3, self.R3, n_pixels)
        A, B = np.meshgrid(x_coords, y_coords, indexing='xy')

        for x, y in np.column_stack([A.ravel(), B.ravel()]):
            x_p, y_p = self.to_pixel_coords((x, y), n_pixels)
            grid[y_p, x_p] = self.pho(x, y)

        return grid

    def to_pixel_coords(self, world_coords, n_pixels):
        x, y = world_coords

        # map [-R3, R3] -> [0, n_pixels-1]
        x_p = int(round((x + self.R3) / (2 * self.R3) * (n_pixels - 1)))
        y_p = int(round((y + self.R3) / (2 * self.R3) * (n_pixels - 1)))

        return x_p, y_p


def partie2(e):
    m = MomentOfInertia(0.4 , 0.1 , 0.8 , 1)
    iters = [int(10**i) for i in range(3,6)]
    for N in iters:
        m.integrationMC(m.g , N , 500)
    pass



if __name__ == "__main__":
    e = Exercice(low = 0 , high= 1 )
    #partie1(e)
    partie2(e)