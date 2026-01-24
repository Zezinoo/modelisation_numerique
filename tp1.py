import numpy as np
from matplotlib import pyplot as plt
import time


class Exercice():

    @staticmethod
    def f(x : np.ndarray) -> np.ndarray:
        return x**2 - 5*np.log(x) - 4
    
    @staticmethod
    def h(x : np.ndarray) -> np.ndarray:
        return np.exp(x)*np.sin(x)
    
    @staticmethod
    def fprime(f , x_k , pas = 1e-6):
        fprime = (f(x_k + pas) - f(x_k))/(pas)
        return fprime


    def plot(self, a,b,points):
        x = np.linspace(a,b,1000)
        iter = [i for i in range(len(points))]

        fig , ax = plt.subplots(2,2 , figsize = (10,4))

        ax[0][0].plot(x , self.f(x))
        ax[0][0].set(title=r"$f(x)$")

        ax[0][1].plot(iter , points)
        ax[0][1].set(title = r"Points vs Iterations")
        ax[0][1].hlines( 3.1102197545478703,  iter[0] , iter[-1] , color = "red" , label = f"x en f(x) = 0.Converge en {len(iter)-1} iterations")
        ax[0][1].legend()

        diffs = np.roll(points , -1)
        diffs = np.abs(points - diffs)[:-1]
        iters = [i for i in range(1,len(points))]
        ax[1][0].semilogy(iters , diffs)

        ax[1][0].set(title = r"Differences vs iterations")
        ax[1][0].grid(True , which = "both")

        plt.show()

    def ex_dichotomie(self , a , b , eps = 1e-12 , plot = True):
        def Dicho(a,b,f,eps):
            a = a + eps # avoid error for functions not defined in zero
            points = []
            x_k = (a + b)/2
            points.append(x_k)
            counter  = 0
            while abs(f(x_k)) >= eps and counter <= 1e3:
                if f(a) * f(x_k) < 0 :
                    b = x_k
                elif f(b) * f(x_k) < 0:
                    a = x_k
                
                x_k = (a+b)/2
                points.append(x_k)
                counter = counter + 1
                if counter > 1e4:
                    raise ValueError(f"Max iterations reached, non converging for parameters chosen : x_k = {x_k}")
            return points


        f = self.f

        start = time.time()
        points = Dicho(a,b,f,eps)
        total = time.time() - start

        if plot : 
            print(f"Exec time : {total}")
            self.plot(a,b,points)

        return total

        


    def ex_newton(self , a , b , xinit , eps = 1e-12 , plot = True):
        fprime = self.fprime    

        def NR(a,b,f,xinit,eps , max_iter = 1e4):
            counter = 0
            points = []
            x_k = xinit
            fprime_k = fprime(f , x_k)
            x_k1 = x_k - f(x_k)/fprime_k
            if x_k1 < a or x_k1 > b:
                    raise ValueError(f"Method doesnt converge : x_k = {x_k1}")
            points.extend([x_k , x_k1])

            while abs(f(x_k1)) > eps and counter < max_iter:
                x_k = x_k1
                fprime_k = fprime(f , x_k)
                x_k1 = x_k - f(x_k)/fprime_k
                points.append(x_k1)
                if x_k1 < a or x_k1 > b:
                    raise ValueError("Method doesnt converge")
                
                counter = counter + 1
            if counter >= max_iter:
                print("Max iter exceeded")
            return points
        
        f = self.f

        start = time.time()
        points = NR(a,b,f,xinit,eps)
        total = time.time() - start

        if plot:
            print(f"Exec time : {total}")
            self.plot(a,b,points)

        return total

        
        # Certaine valeurs de x font avec que les points trouves divergent a cause de la pente trouve

    def ex_integration_numerique(self , f,a,b,N,M):
        def point_milieu(f,a,b,N):
            intervals = np.linspace(a,b,N)
            area = lambda a,b,f :(b-a)* f((a+b)/2)
            total = 0
            for i in range(1,len(intervals)):
                a = intervals[i-1]
                b = intervals[i]
                total = total + area(a,b,f)

            return total

        def trapeze(f,a,b,N) :
            intervals = np.linspace(a,b,N)
            area = lambda a,b,f :(b-a)* (f(a) + f(b))/2
            total = 0
            for i in range(1,len(intervals)):
                a = intervals[i-1]
                b = intervals[i]
                total = total + area(a,b,f)
            return total

        def simpson(f,a,b,N) :
            intervals = np.linspace(a,b,N)
            area = lambda a,b,f :(b-a)/6 * (f(a) + 4*f((a+b)/2) + f(b))
            total = 0
            for i in range(1,len(intervals)):
                a = intervals[i-1]
                b = intervals[i]
                total = total + area(a,b,f)
            return total


        def integrate(x1 , x2 , f, N , M ):
            match M :
                case  "point milieu":
                    I = point_milieu(f,x1,x2,N)
                case "trapeze" :
                    I = trapeze(f,x1,x2,N)
                case "simpson" :
                    I = simpson(f,x1,x2,N)

            return I
        
        start = time.time()
        I = integrate(a , b , f, N , M )
        total = time.time() - start

        return I , total

def partie_1(e):
    a = 1
    b = 10
    eps = 10e-12
    xinit = 2
    e = Exercice()
    
    e = Exercice()
    td = 0
    tn = 0
    for i in range(1000):
        td += e.ex_dichotomie(a,b , eps = eps , plot = False)
        tn += e.ex_newton(a, b, xinit , eps = eps , plot = False) 

    print(f"diff_time {td - tn}" )
    

def partie_2(e):
    N = [int(10**i) for i in range(1,6)]
    a = 0
    b = 9.5

    simps = [e.ex_integration_numerique(e.h , a,b,n,"simpson") for n in N]
    trapeze = [e.ex_integration_numerique(e.h , a,b,n,"trapeze") for n in N]
    rectangle = [e.ex_integration_numerique(e.h , a,b,n,"point milieu") for n in N]


    simps_values = [x[0] for x in simps]
    simps_times = [x[1] for x in simps]

    trapeze_values = [x[0] for x in trapeze]
    trapeze_times = [x[1] for x in trapeze]

    rectangle_values = [x[0] for x in rectangle]
    rectangle_times = [x[1] for x in rectangle]



    exact_area = lambda a,b : 1/2 * (np.exp(b)*(np.sin(b) - np.cos(b)) - np.exp(a)*(np.sin(a) - np.cos(a)))
    relative_error = lambda obs , exact : abs((obs - exact)/exact)

    fig , axs = plt.subplots(2,2 , figsize = (10,4))


    interval = np.linspace(a,b,1000)
    axs[0][0].plot(interval , e.h(interval))
    axs[0][0].set(title = 'f(x)')

    axs[0][1].semilogx(N , simps_values , label = 'simps')
    axs[0][1].semilogx(N , trapeze_values , label = 'trapeze')
    axs[0][1].semilogx(N , rectangle_values , label = 'point milieu')
    axs[0][1].hlines(exact_area(a,b) , 0 , N[-1])
    axs[0][1].set(title = 'values x iterations')

    axs[1][0].semilogx(N , [relative_error(x , exact_area(a,b)) for x in simps_values] , label = 'simps')
    axs[1][0].semilogx(N , [relative_error(x , exact_area(a,b)) for x in trapeze_values] , label = 'trapeze')
    axs[1][0].semilogx(N , [relative_error(x , exact_area(a,b)) for x in rectangle_values] , label = 'point milieu')
    axs[1][0].set(title = 'error x iterations')

    
    axs[1][1].semilogx(N , simps_times , label = 'simps')
    axs[1][1].semilogx(N , trapeze_times , label = 'trapeze')
    axs[1][1].semilogx(N , rectangle_times , label = 'point milieu')
    axs[1][1].set(title = 'time x iterations')

    for ax in axs.flatten():
        ax.legend()

    plt.show()

    f = {
    0: 4.0,
    100: 2.68,
    200: 1.8,
    300: 1.2,
    400: 0.81,
    500: 0.54,
    600: 0.36,
    700: 0.23,
    800: 0.16,
    900: 0.11
    }

    # Redefinindo pq preguica
    def trapeze(f : dict ,a,b,x_values) :
        area = lambda a,b,f :(b-a)* (f.get(a) + f.get(b))/2
        total = 0
        for i in range(1 ,len(x_values)):
            a = x_values[i-1]
            b = x_values[i]
            total = total + area(a,b,f)
        return total
    
    I = trapeze(f , a , b , list(f.keys())) # V*s
    print(I)

    # Integral exacte int_0^inf I0*exp(-t/tau) dt = I0/-tau[0-1] = Io/tau -> tau = I0/Integral

    tau = I/f.get(0)

    print(tau)


if __name__ == "__main__":

    e = Exercice()
    partie_1(e)
    partie_2(e)

       

    def ex_regression(self,p,data):
        from functools import reduce
        def sum_x_of_k_degree(x_arr , k):
            x_arr = [x**k for x in x_arr]
            sum = reduce(lambda x ,y : x+y , x_arr)
            return sum


        def Reg_Poly(data : np.ndarray,p):
            n,_ = data.shape
            x_arr = data[:,0]
            y_arr = data[:,1]
            S = np.array(
                [
                    [sum_x_of_k_degree(x_arr , j) for j in range(i,i+p+1)]
                    for i in range(p+1)]
            )
            W = np.array(
                [reduce(lambda x,y : x +y , np.multiply(x_arr**k , y_arr)) for k in range(p+1) ]
            )
            W = W[:,np.newaxis] 

            C = np.dot(np.linalg.inv(S) , W)

            return C
        
        def Poly(coefficients,p,data):
            x_arr = data[:,0]
            y_values = np.array([coefficients[i][0]*x_arr**i for i in range(p+1)]).sum(axis=0)
            return y_values
        
        coefficients = Reg_Poly(data , p)
        adjustment = Poly(coefficients , p , data)
        
        return adjustment


def partie3(e):
    data = np.array([
    [0,   4.00],
    [80,  2.10],
    [160, 1.15],
    [240, 0.95],
    [320, 0.60],
    [400, 0.42],
    [480, 0.34],
    [560, 0.24],
    [640, 0.19],
    [720, 0.14],
    [800, 0.11]
])

    x_arr = data[:,0]
    y_arr = data[:,1]
    plt.plot(x_arr , y_arr , label = 'Original data')


    for p in [3,5,7]:
        adjustment = e.ex_regression(p,data)
        plt.plot(x_arr , adjustment , label = f"N = {p}")

    plt.legend()
    plt.show()

e = Exercice()

if __name__ == "__main__":
    e = Exercice()
    partie3(e)
    #245 tau