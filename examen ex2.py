import numpy as np
from matplotlib import pyplot as plt
import json

def f(x):
    return np.tan(2*x)

def g(x):
    return np.sqrt(1/(x**2))

a = 1e-10
b = 4

# 1
xs = np.linspace(a,b,1000)

plt.plot(xs , f(xs) , label = "f(x)")
plt.plot(xs , g(xs) , label = "g(x)")

plt.axis([a , b,-5,5])
plt.grid(True)

plt.legend()
plt.show()

def h(x):
    return g(x) - f(x)

def h_prime(x,pas=1e-10):
    return (h(x+pas) - h(x))/pas

plt.plot(xs , h(xs), label = "h(x) = g(x) - f(x)")
plt.axis([a , b,-5,5])
plt.grid(True)

plt.legend()
plt.show()

plt.plot(xs , h_prime(xs) , label = "h(x) prime")
plt.axis([a , b,-1e3,5])
plt.grid(True)

plt.legend()
plt.show()

# On remarque bien 6 points dintersection en linterval (0,4]

# 2
def NR(f,g,x0,x1,x2,epsilon=1e-12):

    if x0 < x1 or x0 > x2:
        raise ValueError("Choose x0 in the (x1,x2) interval")

    def h(x):
        return g(x) - f(x)
    # On cherche le 0 de la fonction h, que donnera les points dintersection
    counter = 0
    def h_prime(x,pas=1e-14):
        return (h(x+pas) - h(x))/pas
    
    diff = 1
    xn = x0
    counter = 0
    
    while diff >= epsilon:
        xn1 = xn - h(xn)/h_prime(xn) # Pas de newton , mise a jour du valeur avec la derivé
        if xn1 < x1 or xn1 > x2: # Si le valeur estime est dehors de l'intervalle , le methode diverge
            raise ValueError("Method doesnt converge for chosen xO")# Erreur pour le cas precedent
        diff = abs(xn - xn1) # Calcul d'erreur entre pas sucessifs
        counter = counter + 1 # Mise a jour du nombre d'iterations
        xn = xn1 # Mise a jour du valeur de xn pour la bocule
        if counter > 100 : # Check si le nombre d'iteration depasse 100
            raise ValueError("Maximum number of iterations exceeded. No zero found for the maximum iterations.") # Erreur pour le nombre max d'iterations


    return xn ,counter


# les valeurs initiales x0 etaient determines pour analyse graphique de la fonction h(x) = g(x) - f(x)

initial_values = [0.52 , 1.8  , 3.2]
zeros = {f"{2*i + 1}th zero" : NR(f,g,initial_values[i],a,b,epsilon=1e-12) for i in range(len(initial_values)) }




# Pour les zeros pairs, on voit bien pour le graphique de hprime(x) que la derivé diverge rapidement. Donc, il faut choisir
# x0 avec soin pour que le methode ne cole pas a les autres 0
# Quand meme , on sait bien que ces point seront a les valeurs ou tan(2x) est infini, c'est a dire
# pi/4 , 3*pi/4 , 5*pi/4

zero2 = NR(f,g,np.pi/4,a,b,epsilon=1e-12)
zero4 = NR(f,g,3*np.pi/4,a,b,epsilon=1e-12)
zero6 = NR(f,g,5*np.pi/4,a,b,epsilon=1e-12)

zeros["2nd zero"] = zero2
zeros["4th zero"] = zero4
zeros["6th zero"] = zero6

# Le output est en forme
"""
  "nth zero": [
    valeur du zero,
    nombre d'iterations requises
"""
print(json.dumps(zeros , indent=2))

# On trace finalement les points d'intersection

xintersection = [zeros[k][0] for k in zeros.keys()]

intersections = [g(xi) for xi in xintersection]


plt.plot(xs , f(xs) , label = "f(x)")
plt.plot(xs , g(xs) , label = "g(x)")
plt.scatter(xintersection , intersections, color="red")

plt.axis([a , b,-5,5])
plt.grid(True)

plt.legend()
plt.show()

plt.plot(xs , h(xs), label = "h(x) = g(x) - f(x)")
plt.scatter(xintersection , [0 for i in range(len(xintersection))], color="red")
plt.axis([a , b,-5,5])
plt.grid(True)

plt.legend()
plt.show()