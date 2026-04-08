"""
TP3 - Méthodes numériques
Résolution d'équations différentielles par différences finies.
Application aux équations d'un laser à solide.

Refactorização orientada ao enunciado do TP:
- nomes de variáveis mais explícitos;
- funções alinhadas às quantidades físicas do PDF;
- séparation claire entre régime stationnaire, Euler et RK4.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np


@dataclass
class ParametresLaser:
    """Paramètres numériques du modèle de laser YAG:Yb3+."""

    lambda_pompage_m: float = 940.0e-9
    lambda_laser_m: float = 1030.0e-9
    concentration_totale_cm3: float = 1.0e20
    duree_vie_s: float = 1.0e-3
    rayon_faisceau_cm: float = 0.01
    longueur_cavite_cm: float = 1.0
    pertes_propagation_cm_inv: float = 0.02
    reflexion_M1: float = 1.0
    reflexion_M2: float = 0.98
    sigma_abs_940_cm2: float = 2.0e-20
    sigma_abs_1030_cm2: float = 0.1e-20
    sigma_em_1030_cm2: float = 0.3e-20
    indice_refraction: float = 1.5
    vitesse_lumiere_m_s: float = 3.0e8
    constante_planck_J_s: float = 6.63e-34
    beta_emission_spontanee_cm_s: float = 1.0e-2

    @property
    def gamma_cm_inv(self) -> float:
        """Coefficient d'atténuation total de la cavité."""
        return self.pertes_propagation_cm_inv - np.log(self.reflexion_M1 * self.reflexion_M2) / (
            2.0 * self.longueur_cavite_cm
        )

    @property
    def surface_faisceau_cm2(self) -> float:
        return np.pi * self.rayon_faisceau_cm**2


class ModeleLaserSolide:
    """Implémente les calculs demandés dans le TP3."""

    def __init__(self, parametres: ParametresLaser | None = None) -> None:
        self.parametres = parametres or ParametresLaser()

    # ------------------------------------------------------------------
    # Conversions puissance <-> flux de photons
    # ------------------------------------------------------------------
    def puissance_vers_flux(self, puissance_W: float, longueur_onde_m: float) -> float:
        p = self.parametres
        return puissance_W * longueur_onde_m / (
            p.surface_faisceau_cm2 * p.constante_planck_J_s * p.vitesse_lumiere_m_s
        )

    def flux_vers_puissance(self, flux_ph_s_cm2: float, longueur_onde_m: float) -> float:
        p = self.parametres
        return flux_ph_s_cm2 * p.surface_faisceau_cm2 * p.constante_planck_J_s * p.vitesse_lumiere_m_s / longueur_onde_m

    # ------------------------------------------------------------------
    # Régime stationnaire
    # ------------------------------------------------------------------
    def population_excitee_stationnaire(self) -> float:
        """Calcule N2 au seuil à partir de l'équation stationnaire du TP."""
        p = self.parametres
        return (p.gamma_cm_inv + p.sigma_abs_1030_cm2 * p.concentration_totale_cm3) / (
            p.sigma_em_1030_cm2 + p.sigma_abs_1030_cm2
        )

    def population_fondamentale_stationnaire(self) -> float:
        p = self.parametres
        return p.concentration_totale_cm3 - self.population_excitee_stationnaire()

    def flux_pompage_seuil(self) -> float:
        p = self.parametres
        N2 = self.population_excitee_stationnaire()
        N1 = p.concentration_totale_cm3 - N2
        return N2 / (p.duree_vie_s * p.sigma_abs_940_cm2 * N1)

    def puissance_pompage_seuil(self) -> float:
        p = self.parametres
        return self.flux_vers_puissance(self.flux_pompage_seuil(), p.lambda_pompage_m)

    def flux_laser_stationnaire(self, puissance_pompage_W: float) -> float:
        p = self.parametres
        N2 = self.population_excitee_stationnaire()
        N1 = p.concentration_totale_cm3 - N2
        flux_pompage = self.puissance_vers_flux(puissance_pompage_W, p.lambda_pompage_m)
        numerateur = 0.5 * (flux_pompage * p.sigma_abs_940_cm2 * N1 - N2 / p.duree_vie_s)
        denominateur = p.sigma_em_1030_cm2 * N2 - p.sigma_abs_1030_cm2 * N1
        return numerateur / denominateur

    def puissance_laser_stationnaire(self, puissance_pompage_W: float) -> float:
        p = self.parametres
        flux_laser = self.flux_laser_stationnaire(puissance_pompage_W)
        return max(self.flux_vers_puissance(flux_laser, p.lambda_laser_m), 0.0)

    # ------------------------------------------------------------------
    # Équations différentielles du modèle
    # ------------------------------------------------------------------
    def derivee_N2(self, N2: float, phi_1030: float, phi_940: float) -> float:
        """Equation (1) du TP."""
        p = self.parametres
        N1 = p.concentration_totale_cm3 - N2
        terme_pompage = phi_940 * p.sigma_abs_940_cm2 * N1
        terme_emission_stimulee = 2.0 * phi_1030 * (p.sigma_em_1030_cm2 * N2 - p.sigma_abs_1030_cm2 * N1)
        terme_relaxation = N2 / p.duree_vie_s
        return terme_pompage - terme_emission_stimulee - terme_relaxation

    def derivee_phi_1030(self, N2: float, phi_1030: float, beta_cm_s: float | None = None) -> float:
        """Equation (2) du TP, avec terme beta pour le régime transitoire."""
        p = self.parametres
        beta = p.beta_emission_spontanee_cm_s if beta_cm_s is None else beta_cm_s
        N1 = p.concentration_totale_cm3 - N2
        vitesse_cm_s = p.vitesse_lumiere_m_s * 100.0
        gain = phi_1030 * (p.sigma_em_1030_cm2 * N2 - p.sigma_abs_1030_cm2 * N1) * vitesse_cm_s / p.indice_refraction
        pertes = p.gamma_cm_inv * phi_1030 * vitesse_cm_s / p.indice_refraction
        source_spontanee = beta * N2 / p.duree_vie_s
        return gain - pertes + source_spontanee

    # ------------------------------------------------------------------
    # Méthodes numériques demandées dans le TP
    # ------------------------------------------------------------------
    def Euler_N2(self, puissance_pompage_W: float, pas_s: float, temps_max_s: float) -> List[Tuple[float, float]]:
        """Méthode d'Euler explicite pour l'équation sur N2 quand phi_1030 = 0."""
        p = self.parametres
        temps = 0.0
        N2 = 0.0
        evolution = []

        while temps < temps_max_s:
            if temps < 3.0e-3:
                phi_940 = self.puissance_vers_flux(puissance_pompage_W, p.lambda_pompage_m)
            else:
                phi_940 = 0.0

            N2_suivant = N2 + pas_s * self.derivee_N2(N2, 0.0, phi_940)
            temps += pas_s
            N2 = N2_suivant
            evolution.append((temps, N2))

        return evolution

    def solution_exacte_N2_pompage_const(self, puissance_pompage_W: float, temps_s: np.ndarray | float) -> np.ndarray | float:
        """Solution exacte sur [0, 3 ms] quand le pompage est constant et phi_1030 = 0."""
        p = self.parametres
        phi_940 = self.puissance_vers_flux(puissance_pompage_W, p.lambda_pompage_m)
        k = phi_940 * p.sigma_abs_940_cm2
        return (k * p.concentration_totale_cm3 * p.duree_vie_s) / (k * p.duree_vie_s + 1.0) * (
            1.0 - np.exp(-(k + 1.0 / p.duree_vie_s) * np.asarray(temps_s))
        )

    def Euler_systeme_laser(self, puissance_pompage_W: float, pas_s: float, temps_max_s: float) -> List[Tuple[float, float, float]]:
        """Euler explicite pour le système couplé (N2, phi_1030)."""
        p = self.parametres
        phi_940 = self.puissance_vers_flux(puissance_pompage_W, p.lambda_pompage_m)
        temps = 0.0
        N2 = 0.0
        phi_1030 = 0.0
        evolution = []

        while temps < temps_max_s:
            N2_suivant = N2 + pas_s * self.derivee_N2(N2, phi_1030, phi_940)
            phi_1030_suivant = phi_1030 + pas_s * self.derivee_phi_1030(N2_suivant, phi_1030)
            temps += pas_s
            N2 = N2_suivant
            phi_1030 = phi_1030_suivant
            evolution.append((temps, N2, phi_1030))

        return evolution

    def _pas_RK4_scalaire(self, fonction: Callable[..., float], y1: float, y2: float, *args: float, pas_s: float) -> float:
        k1 = pas_s * fonction(y1, y2, *args)
        k2 = pas_s * fonction(y1 + pas_s / 2.0, y2 + k1 / 2.0, *args)
        k3 = pas_s * fonction(y1 + pas_s / 2.0, y2 + k2 / 2.0, *args)
        k4 = pas_s * fonction(y1 + pas_s, y2 + k3, *args)
        return (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0

    def RK4_systeme_laser(self, puissance_pompage_W: float, pas_s: float, temps_max_s: float) -> List[Tuple[float, float, float]]:
        """Méthode de Runge-Kutta d'ordre 4 pour le système couplé."""
        p = self.parametres
        phi_940 = self.puissance_vers_flux(puissance_pompage_W, p.lambda_pompage_m)
        temps = 0.0
        N2 = 0.0
        phi_1030 = 0.0
        evolution = []

        while temps < temps_max_s:
            increment_N2 = self._pas_RK4_scalaire(self.derivee_N2, N2, phi_1030, phi_940, pas_s=pas_s)
            N2_suivant = N2 + increment_N2
            increment_phi = self._pas_RK4_scalaire(self.derivee_phi_1030, N2_suivant, phi_1030, pas_s=pas_s)
            phi_1030_suivant = phi_1030 + increment_phi

            temps += pas_s
            N2 = N2_suivant
            phi_1030 = phi_1030_suivant
            evolution.append((temps, N2, phi_1030))

        return evolution

    def RK4_QSwitch(self, puissance_pompage_W: float, pas_s: float, temps_max_s: float, temps_ouverture_s: float = 40.0e-6) -> List[Tuple[float, float, float]]:
        """RK4 avec blocage de l'oscillation laser avant l'ouverture du Q-switch."""
        p = self.parametres
        phi_940 = self.puissance_vers_flux(puissance_pompage_W, p.lambda_pompage_m)
        temps = 0.0
        N2 = 0.0
        phi_1030 = 0.0
        evolution = []

        while temps < temps_max_s:
            increment_N2 = self._pas_RK4_scalaire(self.derivee_N2, N2, phi_1030, phi_940, pas_s=pas_s)
            N2_suivant = N2 + increment_N2

            if temps < temps_ouverture_s:
                phi_1030_suivant = 0.0
            else:
                increment_phi = self._pas_RK4_scalaire(self.derivee_phi_1030, N2_suivant, phi_1030, pas_s=pas_s)
                phi_1030_suivant = phi_1030 + increment_phi

            temps += pas_s
            N2 = N2_suivant
            phi_1030 = phi_1030_suivant
            evolution.append((temps, N2, phi_1030))

        return evolution

    # ------------------------------------------------------------------
    # Outils d'analyse
    # ------------------------------------------------------------------
    @staticmethod
    def largeur_mi_hauteur(signal: np.ndarray, temps: np.ndarray) -> Tuple[int, int, float, float]:
        """Retourne les indices gauche/droite, la demi-hauteur et la largeur FWHM."""
        signal = np.asarray(signal)
        temps = np.asarray(temps)

        maximum = np.max(signal)
        demi_hauteur = maximum / 2.0
        indice_max = int(np.argmax(signal))

        indice_gauche = int(np.argmin(np.abs(signal[:indice_max] - demi_hauteur))) if indice_max > 0 else 0
        indice_droite = int(np.argmin(np.abs(signal[indice_max:] - demi_hauteur)) + indice_max)
        largeur = temps[indice_droite] - temps[indice_gauche]
        return indice_gauche, indice_droite, demi_hauteur, largeur


# ----------------------------------------------------------------------
# Démonstration / exécution directe
# ----------------------------------------------------------------------
if __name__ == "__main__":
    modele = ModeleLaserSolide()
    p = modele.parametres

    # Partie A - régime laser stationnaire
    N2_seuil = modele.population_excitee_stationnaire()
    P940_seuil_W = modele.puissance_pompage_seuil()

    print(f"Population excitée stationnaire N2 = {N2_seuil:.6e} cm^-3")
    print(f"Puissance de pompage au seuil = {P940_seuil_W:.6f} W")

    puissances_pompage = np.linspace(0.0, 10.0, 1000)
    puissances_emises = [(1.0 - p.reflexion_M2) * modele.puissance_laser_stationnaire(P) for P in puissances_pompage]

    plt.plot(puissances_pompage, puissances_emises)
    plt.title("Puissance laser émise en fonction de la puissance de pompage")
    plt.xlabel("P940 (W)")
    plt.ylabel("P_laser émise (W)")
    plt.grid(True)
    plt.show()

    # Partie B - pompage pulsé sans oscillation laser
    evolution_euler_N2 = modele.Euler_N2(puissance_pompage_W=5.0, pas_s=100e-6, temps_max_s=10e-3)
    temps_B = np.array([point[0] for point in evolution_euler_N2])
    N2_B = np.array([point[1] for point in evolution_euler_N2])

    masque_pompage = temps_B <= 3.0e-3
    temps_exact = temps_B[masque_pompage]
    N2_exact = modele.solution_exacte_N2_pompage_const(5.0, temps_exact)

    plt.plot(temps_B, N2_B, label="Euler explicite")
    plt.plot(temps_exact, N2_exact, label="Solution exacte")
    plt.title("Evolution de N2(t)")
    plt.xlabel("t (s)")
    plt.ylabel("N2 (cm^-3)")
    plt.legend()
    plt.grid(True)
    plt.show()

    # Partie C - régime laser transitoire
    evolution_euler_laser = modele.Euler_systeme_laser(puissance_pompage_W=50.0, pas_s=100e-12, temps_max_s=100e-6)
    evolution_rk4_laser = modele.RK4_systeme_laser(puissance_pompage_W=50.0, pas_s=100e-12, temps_max_s=100e-6)

    temps_C = np.array([point[0] for point in evolution_rk4_laser])
    phi_rk4 = np.array([point[2] for point in evolution_rk4_laser])
    phi_euler = np.array([point[2] for point in evolution_euler_laser])
    phi_stationnaire = modele.flux_laser_stationnaire(50.0)

    plt.plot(temps_C, phi_euler, label="Euler")
    plt.plot(temps_C, phi_rk4, label="RK4")
    plt.plot(temps_C, np.ones_like(temps_C) * phi_stationnaire, label="Régime stationnaire")
    plt.title("Evolution du flux laser en régime transitoire")
    plt.xlabel("t (s)")
    plt.ylabel(r"$\Phi_{1030}$ (photons.s$^{-1}$.cm$^{-2}$)")
    plt.legend()
    plt.grid(True)
    plt.show()

    # Partie D - régime Q-switch
    evolution_qswitch = modele.RK4_QSwitch(puissance_pompage_W=50.0, pas_s=1e-9, temps_max_s=100e-6)
    fenetre = [point for point in evolution_qswitch if 40e-6 <= point[0] <= 40e-6 + 100e-9]
    temps_D = np.array([point[0] for point in fenetre])
    phi_D = np.array([point[2] for point in fenetre])
    puissance_D = np.array([modele.flux_vers_puissance(phi, p.lambda_laser_m) for phi in phi_D])

    indice_gauche, indice_droite, demi_hauteur, fwhm = modele.largeur_mi_hauteur(puissance_D, temps_D)

    plt.plot(temps_D, puissance_D, label="Impulsion Q-switch")
    plt.hlines(demi_hauteur, temps_D[indice_gauche], temps_D[indice_droite], colors="red", label=f"FWHM = {fwhm:.2e} s")
    plt.title("Impulsion laser déclenchée")
    plt.xlabel("t (s)")
    plt.ylabel("Puissance (W)")
    plt.legend()
    plt.grid(True)
    plt.show()
