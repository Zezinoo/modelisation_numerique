import time
from typing import Callable

import numpy as np
from matplotlib import pyplot as plt


class Exercice:
    """Implémentation des exercices du TP1 de modélisation numérique."""

    @staticmethod
    def f(x: np.ndarray | float) -> np.ndarray | float:
        """Fonction de la partie I : x^2 - 5 ln(x) - 4."""
        return x**2 - 5 * np.log(x) - 4

    @staticmethod
    def h(x: np.ndarray | float) -> np.ndarray | float:
        """Fonction de la partie II : exp(x) sin(x)."""
        return np.exp(x) * np.sin(x)

    @staticmethod
    def fprime(f: Callable[[float], float], x_k: float, pas: float = 1e-6) -> float:
        """Approximation numérique de la dérivée de f en x_k."""
        return (f(x_k + pas) - f(x_k)) / pas

    def plot_resolution(self, a: float, b: float, x_values: list[float]) -> None:
        """Trace les graphiques demandés pour la résolution d'équation."""
        x_plot = np.linspace(a, b, 1000)
        iterations = list(range(len(x_values)))

        fig, axes = plt.subplots(2, 2, figsize=(10, 4))

        axes[0][0].plot(x_plot, self.f(x_plot))
        axes[0][0].set(title=r"$f(x)$")

        axes[0][1].plot(iterations, x_values)
        axes[0][1].set(title="Évolution de x selon le nombre d'itérations")
        axes[0][1].hlines(
            3.1102197545478703,
            iterations[0],
            iterations[-1],
            color="red",
            label=f"racine approchée, convergence en {len(iterations) - 1} itérations",
        )
        axes[0][1].legend()

        differences = np.abs(np.diff(x_values))
        difference_iterations = list(range(1, len(x_values)))
        axes[1][0].semilogy(difference_iterations, differences)
        axes[1][0].set(title=r"Évolution de $|x^{(k+1)} - x^{(k)}|$")
        axes[1][0].grid(True, which="both")

        axes[1][1].axis("off")
        plt.tight_layout()
        plt.show()

    def ex_dichotomie(self, a: float, b: float, eps: float = 1e-12, plot: bool = True) -> float:
        """Mesure le temps d'exécution de la méthode de dichotomie."""

        def Dicho(a: float, b: float, f: Callable[[float], float], epsilon: float) -> list[float]:
            """Méthode de dichotomie demandée dans l'énoncé.

            Retourne la liste des valeurs successives x^(k).
            """
            borne_gauche = a + epsilon  # évite log(0) si l'intervalle commence en 0
            borne_droite = b
            x_values = []
            x_k = (borne_gauche + borne_droite) / 2
            x_values.append(x_k)
            iteration = 0
            max_iter = 1000

            while abs(f(x_k)) >= epsilon and iteration <= max_iter:
                if f(borne_gauche) * f(x_k) < 0:
                    borne_droite = x_k
                elif f(borne_droite) * f(x_k) < 0:
                    borne_gauche = x_k

                x_k = (borne_gauche + borne_droite) / 2
                x_values.append(x_k)
                iteration += 1

                if iteration > max_iter:
                    raise ValueError(
                        f"Nombre maximal d'itérations atteint : x_k = {x_k}"
                    )

            return x_values

        start_time = time.time()
        x_values = Dicho(a, b, self.f, eps)
        elapsed_time = time.time() - start_time

        if plot:
            print(f"Temps d'exécution : {elapsed_time}")
            self.plot_resolution(a, b, x_values)

        return elapsed_time

    def ex_newton(
        self,
        a: float,
        b: float,
        xinit: float,
        eps: float = 1e-12,
        plot: bool = True,
    ) -> float:
        """Mesure le temps d'exécution de la méthode de Newton-Raphson."""

        def NR(
            a: float,
            b: float,
            f: Callable[[float], float],
            xinit: float,
            epsilon: float,
            max_iter: int = 10_000,
        ) -> list[float]:
            """Méthode de Newton-Raphson demandée dans l'énoncé.

            Retourne la liste des valeurs successives x^(k).
            """
            x_values = []
            iteration = 0
            x_k = xinit

            fprime_k = self.fprime(f, x_k)
            x_k1 = x_k - f(x_k) / fprime_k

            if x_k1 < a or x_k1 > b:
                raise ValueError(f"La méthode ne converge pas : x_k = {x_k1}")

            x_values.extend([x_k, x_k1])

            while abs(f(x_k1)) > epsilon and iteration < max_iter:
                x_k = x_k1
                fprime_k = self.fprime(f, x_k)
                x_k1 = x_k - f(x_k) / fprime_k
                x_values.append(x_k1)

                if x_k1 < a or x_k1 > b:
                    raise ValueError("La méthode ne converge pas sur l'intervalle donné")

                iteration += 1

            if iteration >= max_iter:
                print("Nombre maximal d'itérations dépassé")

            return x_values

        start_time = time.time()
        x_values = NR(a, b, self.f, xinit, eps)
        elapsed_time = time.time() - start_time

        if plot:
            print(f"Temps d'exécution : {elapsed_time}")
            self.plot_resolution(a, b, x_values)

        return elapsed_time

    def ex_integration_numerique(
        self,
        f: Callable[[float], float],
        a: float,
        b: float,
        N: int,
        M: str,
    ) -> tuple[float, float]:
        """Mesure le temps d'exécution de l'intégration numérique."""

        def point_milieu(f: Callable[[float], float], a: float, b: float, N: int) -> float:
            bornes = np.linspace(a, b, N)
            integrale = 0.0
            for i in range(1, len(bornes)):
                x_gauche = bornes[i - 1]
                x_droite = bornes[i]
                integrale += (x_droite - x_gauche) * f((x_gauche + x_droite) / 2)
            return integrale

        def trapeze(f: Callable[[float], float], a: float, b: float, N: int) -> float:
            bornes = np.linspace(a, b, N)
            integrale = 0.0
            for i in range(1, len(bornes)):
                x_gauche = bornes[i - 1]
                x_droite = bornes[i]
                integrale += (x_droite - x_gauche) * (f(x_gauche) + f(x_droite)) / 2
            return integrale

        def simpson(f: Callable[[float], float], a: float, b: float, N: int) -> float:
            bornes = np.linspace(a, b, N)
            integrale = 0.0
            for i in range(1, len(bornes)):
                x_gauche = bornes[i - 1]
                x_droite = bornes[i]
                integrale += (
                    (x_droite - x_gauche)
                    / 6
                    * (f(x_gauche) + 4 * f((x_gauche + x_droite) / 2) + f(x_droite))
                )
            return integrale

        def integrate(x1: float, x2: float, f: Callable[[float], float], N: int, M: str) -> float:
            """Fonction générique demandée dans l'énoncé."""
            match M:
                case "point milieu":
                    return point_milieu(f, x1, x2, N)
                case "trapeze":
                    return trapeze(f, x1, x2, N)
                case "simpson":
                    return simpson(f, x1, x2, N)
                case _:
                    raise ValueError(f"Méthode inconnue : {M}")

        start_time = time.time()
        integrale = integrate(a, b, f, N, M)
        elapsed_time = time.time() - start_time
        return integrale, elapsed_time

    def ex_regression(self, p: int, data: np.ndarray) -> np.ndarray:
        """Calcule l'ajustement polynomial pour les données fournies."""

        def somme_puissance_x(x_values: np.ndarray, k: int) -> float:
            return np.sum(x_values**k)

        def Reg_Poly(data: np.ndarray, p: int) -> np.ndarray:
            """Fonction demandée dans l'énoncé.

            Retourne les p+1 coefficients du polynôme d'ajustement.
            """
            x_values = data[:, 0]
            y_values = data[:, 1]

            S = np.array(
                [
                    [somme_puissance_x(x_values, j) for j in range(i, i + p + 1)]
                    for i in range(p + 1)
                ]
            )

            W = np.array([
                np.sum((x_values**k) * y_values) for k in range(p + 1)
            ])[:, np.newaxis]

            C = np.dot(np.linalg.inv(S), W)
            return C

        def Poly(x: np.ndarray, p: int, data: np.ndarray) -> np.ndarray:
            """Fonction demandée dans l'énoncé.

            Retourne les valeurs du polynôme ajusté aux abscisses x.
            """
            coefficients = Reg_Poly(data, p)
            return np.sum([coefficients[i][0] * x**i for i in range(p + 1)], axis=0)

        x_values = data[:, 0]
        return Poly(x_values, p, data)


def partie_1(exercice: Exercice) -> None:
    a = 1
    b = 10
    eps = 1e-11
    xinit = 2

    temps_dichotomie = 0.0
    temps_newton = 0.0

    for _ in range(1000):
        temps_dichotomie += exercice.ex_dichotomie(a, b, eps=eps, plot=False)
        temps_newton += exercice.ex_newton(a, b, xinit, eps=eps, plot=False)

    print(f"Différence de temps : {temps_dichotomie - temps_newton}")


def partie_2(exercice: Exercice) -> None:
    N_values = [int(10**i) for i in range(1, 6)]
    a = 0
    b = 9.5

    resultats_simpson = [
        exercice.ex_integration_numerique(exercice.h, a, b, N, "simpson") for N in N_values
    ]
    resultats_trapeze = [
        exercice.ex_integration_numerique(exercice.h, a, b, N, "trapeze") for N in N_values
    ]
    resultats_point_milieu = [
        exercice.ex_integration_numerique(exercice.h, a, b, N, "point milieu")
        for N in N_values
    ]

    simpson_values = [resultat[0] for resultat in resultats_simpson]
    simpson_times = [resultat[1] for resultat in resultats_simpson]

    trapeze_values = [resultat[0] for resultat in resultats_trapeze]
    trapeze_times = [resultat[1] for resultat in resultats_trapeze]

    point_milieu_values = [resultat[0] for resultat in resultats_point_milieu]
    point_milieu_times = [resultat[1] for resultat in resultats_point_milieu]

    exact_area = lambda x1, x2: 0.5 * (
        np.exp(x2) * (np.sin(x2) - np.cos(x2))
        - np.exp(x1) * (np.sin(x1) - np.cos(x1))
    )
    relative_error = lambda valeur_obtenue, valeur_exacte: abs(
        (valeur_obtenue - valeur_exacte) / valeur_exacte
    )

    fig, axes = plt.subplots(2, 2, figsize=(10, 4))

    intervalle = np.linspace(a, b, 1000)
    axes[0][0].plot(intervalle, exercice.h(intervalle))
    axes[0][0].set(title="f(x)")

    axes[0][1].semilogx(N_values, simpson_values, label="simpson")
    axes[0][1].semilogx(N_values, trapeze_values, label="trapeze")
    axes[0][1].semilogx(N_values, point_milieu_values, label="point milieu")
    axes[0][1].hlines(exact_area(a, b), 0, N_values[-1])
    axes[0][1].set(title="Résultat de l'intégration numérique")

    axes[1][0].semilogx(
        N_values,
        [relative_error(valeur, exact_area(a, b)) for valeur in simpson_values],
        label="simpson",
    )
    axes[1][0].semilogx(
        N_values,
        [relative_error(valeur, exact_area(a, b)) for valeur in trapeze_values],
        label="trapeze",
    )
    axes[1][0].semilogx(
        N_values,
        [relative_error(valeur, exact_area(a, b)) for valeur in point_milieu_values],
        label="point milieu",
    )
    axes[1][0].set(title="Erreur relative")

    axes[1][1].semilogx(N_values, simpson_times, label="simpson")
    axes[1][1].semilogx(N_values, trapeze_times, label="trapeze")
    axes[1][1].semilogx(N_values, point_milieu_times, label="point milieu")
    axes[1][1].set(title="Temps de calcul")

    for axis in axes.flatten():
        axis.legend()

    plt.tight_layout()
    plt.show()

    mesures = {
        0: 4.0,
        100: 2.68,
        200: 1.8,
        300: 1.2,
        400: 0.81,
        500: 0.54,
        600: 0.36,
        700: 0.23,
        800: 0.16,
        900: 0.11,
    }

    def trapeze_experimental(valeurs: dict[int, float], temps: list[int]) -> float:
        integrale = 0.0
        for i in range(1, len(temps)):
            t_gauche = temps[i - 1]
            t_droite = temps[i]
            integrale += (t_droite - t_gauche) * (
                valeurs[t_gauche] + valeurs[t_droite]
            ) / 2
        return integrale

    integrale_experimentale = trapeze_experimental(mesures, list(mesures.keys()))
    print(integrale_experimentale)

    tau = integrale_experimentale / mesures[0]
    print(tau)


def partie_3(exercice: Exercice) -> None:
    data = np.array(
        [
            [0, 4.00],
            [80, 2.10],
            [160, 1.15],
            [240, 0.95],
            [320, 0.60],
            [400, 0.42],
            [480, 0.34],
            [560, 0.24],
            [640, 0.19],
            [720, 0.14],
            [800, 0.11],
        ]
    )

    x_values = data[:, 0]
    y_values = data[:, 1]
    plt.plot(x_values, y_values, label="Données expérimentales")

    for degre in [3, 5, 7]:
        ajustement = exercice.ex_regression(degre, data)
        plt.plot(x_values, ajustement, label=f"p = {degre}")

    plt.legend()
    plt.show()


if __name__ == "__main__":
    exercice = Exercice()
    partie_1(exercice)
    partie_2(exercice)
    partie_3(exercice)
