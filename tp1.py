import numpy as np
from matplotlib import pyplot as plt
import time


class Exercice():

    @staticmethod
    def f(x : np.ndarray) -> np.ndarray:
        return x**2 - 5*np.log(x) - 4
    

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

        plt.legend()
        plt.show()

    def ex_dichotomie(self):
        def Dicho(a,b,f,eps):
            points = []
            x_k = (a + b)/2
            x_k1 = 0
            points.append(x_k)
            while abs(f(x_k)) >= eps:
                x_k = x_k = (a + b)/2
                if f(a) * f(x_k) < 0 :
                    b = x_k
                elif f:
                    a = x_k
                
                x_k1 = (a+b)/2
                points.append(x_k1)
            return points


        f = self.f

        a = 1
        b = 5
        eps = 10e-12

        start = time.time()
        points = Dicho(a,b,f,eps)
        total = time.time() - start

        print(f"Exec time : {total}")
        self.plot(a,b,points)

        return total

        


    def ex_newton(self):
        def fprime(f , x_k , pas = 1e-6):
            fprime = (f(x_k + pas) - f(x_k))/(pas)
            return fprime

        def NR(a,b,f,xinit,eps , max_iter = 1e4):
            counter = 0
            points = []
            x_k = xinit
            fprime_k = fprime(f , x_k)
            x_k1 = x_k - f(x_k)/fprime_k
            points.extend([x_k , x_k1])
            while f(x_k1) > eps and counter < max_iter:
                x_k = x_k1
                print(x_k1)
                fprime_k = fprime(f , x_k)
                x_k1 = x_k - f(x_k)/fprime_k
                points.append(x_k1)
                if x_k1 < a or x_k1 > b:
                    raise ValueError("Method doesent converge")
                
                counter = counter + 1
            if counter >= max_iter:
                print("Max iter exceeded")
            return points
        
        f = self.f

        a = 0
        b = 10
        eps = 10e-12
        xinit = 2.0
        start = time.time()
        points = NR(a,b,f,xinit,eps)
        total = time.time() - start

        print(f"Exec time : {total}")
        self.plot(a,b,points)

        return total

        
        # Certaine valeurs de x font avec que les points trouves divergent a cause de la pente trouve

    def ex_integration_numerique():
        def point_milieu(f,a,b,N):
            intervals = np.linspace(a,b,f,N)
            area = lambda a,b,f :(b-a)* f((a+b)/2)
            total = 0
            for i in range(1,len(intervals)):
                a = intervals[i-1]
                b = intervals[i]
                total = total + area(a,b,f)

            return total

        def trapeze(f,a,b,N) :
            intervals = np.linspace(a,b,f,N)
            area = lambda a,b,f :(b-a)* (f(a) + f(b))/2
            total = 0
            for i in range(1,len(intervals)):
                a = intervals[i-1]
                b = intervals[i]
                total = total + area(a,b,f)
            return total

        def simpson(f,a,b,N) :
            intervals = np.linspace(a,b,f,N)
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
                    I = point_milieu(x1,x2,f,N)
                case "trapeze" :
                    I = trapeze(x1,x2,f,N)
                case "simpson" :
                    I = simpson(x1,x2,f,N)

        

e = Exercice()
td = e.ex_dichotomie()
tn = e.ex_newton()

print(f"diff_time {td - tn}" )

