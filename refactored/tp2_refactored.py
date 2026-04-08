import numpy as np
from matplotlib import pyplot as plt


class ExerciceMonteCarlo:
    """
    Implémentation refactorée du TP2 - Méthodes de Monte Carlo.
    L'objectif principal ici est d'aligner les noms des fonctions et des
    variables avec l'énoncé du PDF.
    """

    def __init__(self, a=0.0, b=1.0):
        self.a = a
        self.b = b

    def g(self, x):
        """Fonction g(x) de la partie II."""
        return 2 * (np.exp(x) - 1) / (np.e - 1)

    def primitive_g(self, x):
        """Primitive de g(x) pour calculer l'intégrale exacte."""
        return 2 / (np.e - 1) * (np.exp(x) - x)

    def Iexacte(self):
        """Valeur exacte de l'intégrale de g sur [a, b]."""
        return self.primitive_g(self.b) - self.primitive_g(self.a)

    @staticmethod
    def densite_uniforme(x):
        """Densité uniforme sur [0, 1]."""
        return np.ones_like(x, dtype=float)

    @staticmethod
    def densite_2x(x):
        """Densité f(x) = 2x sur [0, 1]."""
        return 2 * x

    @staticmethod
    def inverse_repartition_2x(u):
        """Si F(x) = x², alors F^-1(u) = sqrt(u)."""
        return np.sqrt(u)

    @staticmethod
    def calculer_taille_echantillon(ecart_type, u_alpha, erreur_relative, valeur_exacte):
        """
        Calcule n pour garantir une erreur relative cible à partir de l'IC.
        """
        return int(np.ceil((u_alpha * ecart_type / (erreur_relative * valeur_exacte)) ** 2))

    @staticmethod
    def variance_echantillon(valeurs):
        """
        Estimation ponctuelle de sigma_h^2 à partir de l'échantillon.
        """
        moyenne = np.mean(valeurs)
        moyenne_carre = np.mean(valeurs ** 2)
        return moyenne_carre - moyenne ** 2

    def integrationMC1(self, fonction, a, b, n):
        """
        Question II.3
        Intégration Monte Carlo avec une loi uniforme sur [a, b].
        Retourne l'estimation de l'intégrale et l'estimation de sigma_h^2.
        """
        echantillon = np.random.uniform(low=a, high=b, size=n)

        # Pour une loi uniforme sur [a, b], f(x) = 1 / (b - a)
        densite = 1.0 / (b - a)
        h_values = fonction(echantillon) / densite

        estimation_integrale = np.mean(h_values)
        estimation_sigma_h2 = self.variance_echantillon(h_values)

        return estimation_integrale, estimation_sigma_h2

    def integrationMC2(self, fonction, a, b, n):
        """
        Question II.5
        Intégration Monte Carlo avec la densité f(x) = 2x sur [0, 1].
        """
        u = np.random.uniform(low=0.0, high=1.0, size=n)
        echantillon = self.inverse_repartition_2x(u)

        densite = self.densite_2x(echantillon)
        h_values = fonction(echantillon) / densite

        estimation_integrale = np.mean(h_values)
        estimation_sigma_h2 = self.variance_echantillon(h_values)

        return estimation_integrale, estimation_sigma_h2

    def tracer_evolution_integrale(self, n_values):
        """
        Trace l'évolution de l'estimation de I pour integrationMC1.
        """
        estimations = [self.integrationMC1(self.g, self.a, self.b, n)[0] for n in n_values]
        valeur_exacte = self.Iexacte()

        plt.figure()
        plt.plot(n_values, estimations, label="Monte Carlo uniforme")
        plt.hlines(valeur_exacte, n_values[0], n_values[-1], colors="red", label="I exacte")
        plt.xlabel("n")
        plt.ylabel("I")
        plt.legend()
        plt.grid(True)
        plt.show()

    def tracer_comparaison_erreurs(self, n_values):
        """
        Compare les erreurs relatives pour integrationMC1 et integrationMC2.
        """
        valeur_exacte = self.Iexacte()

        erreurs_mc1 = [
            abs((self.integrationMC1(self.g, self.a, self.b, n)[0] - valeur_exacte) / valeur_exacte)
            for n in n_values
        ]
        erreurs_mc2 = [
            abs((self.integrationMC2(self.g, self.a, self.b, n)[0] - valeur_exacte) / valeur_exacte)
            for n in n_values
        ]

        plt.figure()
        plt.plot(n_values, erreurs_mc1, label="integrationMC1 : uniforme")
        plt.plot(n_values, erreurs_mc2, label="integrationMC2 : f(x)=2x")
        plt.xlabel("n")
        plt.ylabel("Erreur relative")
        plt.legend()
        plt.grid(True)
        plt.show()

    def tracer_densites(self):
        """
        Superpose g, la densité uniforme et la densité 2x.
        """
        x = np.linspace(self.a, self.b, 1000)

        plt.figure()
        plt.plot(x, self.g(x), label="g(x)")
        plt.plot(x, np.ones_like(x), label="f uniforme")
        plt.plot(x, self.densite_2x(x), label="f(x)=2x")
        plt.xlabel("x")
        plt.legend()
        plt.grid(True)
        plt.show()


class MomentInertieMonteCarlo:
    """
    Partie III - Calcul de moments d'inertie par intégration Monte Carlo.
    param_objet est de la forme [R1, R2, a].
    Avec R3 = 1 dans l'énoncé.
    """

    def __init__(self, param_objet, R3=1.0):
        self.R1, self.R2, self.a = param_objet
        self.R3 = R3
        self.x_min = -R3
        self.x_max = R3
        self.y_min = -R3
        self.y_max = R3

    def rho(self, x, y, param_objet=None):
        """
        Question III.1
        Retourne rho(x, y), donc 0 ou 1.
        """
        if param_objet is not None:
            R1, R2, a = param_objet
        else:
            R1, R2, a = self.R1, self.R2, self.a

        rayon_carre = x ** 2 + y ** 2

        dans_couronne_externe = rayon_carre <= self.R3 ** 2 and rayon_carre >= R2 ** 2
        hors_trou_central = rayon_carre >= R1 ** 2
        dans_bras_horizontal = abs(y) <= a and abs(x) <= R2
        dans_bras_vertical = abs(x) <= a and abs(y) <= R2
        dans_croix = hors_trou_central and (dans_bras_horizontal or dans_bras_vertical)

        return 1 if (dans_couronne_externe or dans_croix) else 0

    def g(self, x, y, param_objet=None):
        """
        Question III.1
        g(x, y) = (x² + y²) * rho(x, y)
        """
        return (x ** 2 + y ** 2) * self.rho(x, y, param_objet)

    def creer_image_objet(self, npixels):
        """
        Crée l'image carrée de l'objet.
        """
        image = np.zeros((npixels, npixels), dtype=float)
        x_values = np.linspace(-self.R3, self.R3, npixels)
        y_values = np.linspace(-self.R3, self.R3, npixels)

        for indice_y, y in enumerate(y_values):
            for indice_x, x in enumerate(x_values):
                image[indice_y, indice_x] = self.rho(x, y)

        return image

    def coords_vers_pixels(self, x, y, npixels):
        """
        Convertit des coordonnées réelles en indices image.
        """
        pixel_x = int(round((x + self.R3) / (2 * self.R3) * (npixels - 1)))
        pixel_y = int(round((y + self.R3) / (2 * self.R3) * (npixels - 1)))

        pixel_x = np.clip(pixel_x, 0, npixels - 1)
        pixel_y = np.clip(pixel_y, 0, npixels - 1)

        return pixel_x, pixel_y

    def integrationMC2D1(self, fonction, param_objet, n, npixels):
        """
        Question III.2
        Échantillonnage uniforme sur le carré Ω.
        Retourne :
            - estimation de Jz
            - estimation de sigma_h^2
            - image carrée npixels x npixels
        """
        x_samples = np.random.uniform(self.x_min, self.x_max, size=n)
        y_samples = np.random.uniform(self.y_min, self.y_max, size=n)

        aire_omega = (self.x_max - self.x_min) * (self.y_max - self.y_min)
        densite_uniforme = 1.0 / aire_omega

        h_values = np.array([
            fonction(x, y, param_objet) / densite_uniforme
            for x, y in zip(x_samples, y_samples)
        ])

        estimation_Jz = np.mean(h_values)
        estimation_sigma_h2 = np.mean(h_values ** 2) - estimation_Jz ** 2

        image = self.creer_image_objet(npixels)
        for x, y in zip(x_samples, y_samples):
            if self.rho(x, y, param_objet) == 1:
                pixel_x, pixel_y = self.coords_vers_pixels(x, y, npixels)
                image[pixel_y, pixel_x] = 1.0

        return estimation_Jz, estimation_sigma_h2, image

    def integrationMC2D2(self, fonction, param_objet, n, npixels):
        """
        Version correspondant à la question III.4 :
        échantillonnage sur le disque unité avec densité proportionnelle à x²+y².
        En polaires : f(r, theta) = 2r² / pi
        avec theta = 2pi U1 et r = U2^(1/4)
        """
        u1 = np.random.uniform(0.0, 1.0, size=n)
        u2 = np.random.uniform(0.0, 1.0, size=n)

        theta_samples = 2 * np.pi * u1
        r_samples = u2 ** 0.25

        x_samples = r_samples * np.cos(theta_samples)
        y_samples = r_samples * np.sin(theta_samples)

        def densite(x, y):
            return 2 * (x ** 2 + y ** 2) / np.pi

        h_values = np.array([
            fonction(x, y, param_objet) / densite(x, y)
            for x, y in zip(x_samples, y_samples)
        ])

        estimation_Jz = np.mean(h_values)
        estimation_sigma_h2 = np.mean(h_values ** 2) - estimation_Jz ** 2

        image = self.creer_image_objet(npixels)
        for x, y in zip(x_samples, y_samples):
            if self.rho(x, y, param_objet) == 1:
                pixel_x, pixel_y = self.coords_vers_pixels(x, y, npixels)
                image[pixel_y, pixel_x] = 1.0

        return estimation_Jz, estimation_sigma_h2, image

    @staticmethod
    def intervalle_confiance_95(estimation, sigma_h2, n):
        demi_largeur = 1.96 * np.sqrt(sigma_h2 / n)
        return estimation - demi_largeur, estimation + demi_largeur

    @staticmethod
    def afficher_image(image, titre="Image de l'échantillon"):
        plt.figure()
        plt.imshow(image, cmap="gray", origin="lower")
        plt.title(titre)
        plt.show()


def partie_2_integrale():
    exercice = ExerciceMonteCarlo(a=0.0, b=1.0)

    valeur_exacte = exercice.Iexacte()
    print("I exacte =", valeur_exacte)

    # Approximation de sigma_h^2 avec une discrétisation fine dans le cas uniforme
    x = np.linspace(0.0, 1.0, 100000)
    h_values = exercice.g(x)  # car densité uniforme sur [0, 1] vaut 1
    sigma_h2_uniforme = np.mean(h_values ** 2) - valeur_exacte ** 2

    n_optimal = exercice.calculer_taille_echantillon(
        ecart_type=np.sqrt(sigma_h2_uniforme),
        u_alpha=1.96,
        erreur_relative=0.01,
        valeur_exacte=valeur_exacte,
    )
    print("n pour 1% à 95% =", n_optimal)

    estimation_mc1, variance_mc1 = exercice.integrationMC1(exercice.g, 0.0, 1.0, n_optimal)
    estimation_mc2, variance_mc2 = exercice.integrationMC2(exercice.g, 0.0, 1.0, n_optimal)

    print("integrationMC1 :", estimation_mc1, variance_mc1)
    print("integrationMC2 :", estimation_mc2, variance_mc2)

    n_values = list(range(100, 100001, 100))
    exercice.tracer_evolution_integrale(n_values)
    exercice.tracer_densites()
    exercice.tracer_comparaison_erreurs(n_values)


def partie_3_moment_inertie():
    param_objet = [0.1, 0.8, 0.15]  # [R1, R2, a]
    moment = MomentInertieMonteCarlo(param_objet=param_objet, R3=1.0)

    for n in [1000, 10000, 100000]:
        Jz, sigma_h2, image = moment.integrationMC2D1(moment.g, param_objet, n, npixels=500)
        ic95 = moment.intervalle_confiance_95(Jz, sigma_h2, n)

        print(f"Uniforme - n={n}")
        print("Jz =", Jz)
        print("sigma_h^2 =", sigma_h2)
        print("IC 95% =", ic95)

        moment.afficher_image(image, titre=f"Échantillonnage uniforme - n={n}")

    for n in [1000, 10000, 100000]:
        Jz, sigma_h2, image = moment.integrationMC2D2(moment.g, param_objet, n, npixels=500)
        ic95 = moment.intervalle_confiance_95(Jz, sigma_h2, n)

        print(f"Disque pondéré - n={n}")
        print("Jz =", Jz)
        print("sigma_h^2 =", sigma_h2)
        print("IC 95% =", ic95)

        moment.afficher_image(image, titre=f"Échantillonnage pondéré - n={n}")


if __name__ == "__main__":
    partie_2_integrale()
    partie_3_moment_inertie()
