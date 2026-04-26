import numpy as np
from matplotlib import pyplot as plt


# Constants
alpha = 0.5  # cm-1
L = 6  # cm
N_PHOTONS = 100_000


# Questions

# 1
"""
If Z is the random propagation length before an interaction, it follows an
exponential pdf:

f_Z(z) = alpha * exp(-alpha*z)

The CDF is:

F_Z(z) = integral_0^z alpha * exp(-alpha*u) du
       = 1 - exp(-alpha*z)

This is the same law as Beer-Lambert attenuation:

P(Z > z) = exp(-alpha*z)
"""

# 2
"""
Inverse transform sampling:

u = F_Z(z) = 1 - exp(-alpha*z)
exp(-alpha*z) = 1 - u
z = -ln(1 - u) / alpha

So if U follows a uniform law on [0, 1], then

Z = F_Z^{-1}(U)

follows the exponential law.
"""

# 4
"""
For isotropic diffusion, the solid angle element is:

dOmega = sin(theta) dtheta dphi

After integrating over phi:

F(theta) = (1 / 2) * integral_0^theta sin(t) dt
         = (1 / 2) * (1 - cos(theta))

Therefore:

u = (1 / 2) * (1 - cos(theta))
cos(theta) = 1 - 2u
"""


class Photon:
    def __init__(self, x, y, z, costheta, phi, n_diffusions):
        self.x = x
        self.y = y
        self.z = z
        self.costheta = costheta
        self.phi = phi
        self.current_diffusion = n_diffusions


def ZCDF(z):
    return 1 - np.exp(-alpha * z)


def inverse_ZCDF(u):
    return -np.log(1 - u) / alpha


def thetaCDF(theta):
    return 0.5 * (1 - np.cos(theta))


def inverse_costhetaCDF(u):
    return 1 - 2 * u


def spherical_to_cartesian(r, costheta, phi):
    sintheta = np.sqrt(1 - costheta**2)
    x = r * sintheta * np.cos(phi)
    y = r * sintheta * np.sin(phi)
    z = r * costheta

    return x, y, z


def diffusion(n_photons):
    exited_photons = []
    photons_left = n_photons
    n_diffusions = 0

    # Photons start at the entrance of the slab, in a perfectly collimated beam.
    in_photons_x = np.zeros(n_photons)
    in_photons_y = np.zeros(n_photons)
    in_photons_z = np.zeros(n_photons)

    while photons_left > 0:
        r_samples = inverse_ZCDF(np.random.uniform(0, 1, photons_left))

        if n_diffusions == 0:
            costheta_samples = np.ones(photons_left)
        else:
            costheta_samples = inverse_costhetaCDF(
                np.random.uniform(0, 1, photons_left)
            )

        phi_samples = np.random.uniform(0, 2 * np.pi, photons_left)
        dx, dy, dz = spherical_to_cartesian(r_samples, costheta_samples, phi_samples)

        next_x = in_photons_x + dx
        next_y = in_photons_y + dy
        next_z = in_photons_z + dz

        mask_out = (next_z <= 0) | (next_z >= L)

        if np.any(mask_out):
            old_x = in_photons_x[mask_out]
            old_y = in_photons_y[mask_out]
            old_z = in_photons_z[mask_out]
            out_costheta = costheta_samples[mask_out]
            out_phi = phi_samples[mask_out]

            boundaries = np.where(next_z[mask_out] >= L, L, 0)

            # Stop at the slab surface instead of keeping the overshoot distance.
            distances_to_boundary = (boundaries - old_z) / out_costheta
            exit_dx, exit_dy, _ = spherical_to_cartesian(
                distances_to_boundary, out_costheta, out_phi
            )

            for x, y, z, costheta, phi in zip(
                old_x + exit_dx,
                old_y + exit_dy,
                boundaries,
                out_costheta,
                out_phi,
            ):
                exited_photons.append(
                    Photon(x, y, z, costheta, phi, n_diffusions)
                )

        in_photons_x = next_x[~mask_out]
        in_photons_y = next_y[~mask_out]
        in_photons_z = next_z[~mask_out]

        photons_left = len(in_photons_z)
        n_diffusions += 1

    return exited_photons


def plot_results(forward_photons, backward_photons):
    x_forward = [p.x for p in forward_photons]
    y_forward = [p.y for p in forward_photons]
    x_backward = [p.x for p in backward_photons]
    y_backward = [p.y for p in backward_photons]

    fig, axs = plt.subplots(2, 2)

    axs[0][0].hist(x_forward, bins=100, label="x forward")
    axs[0][1].hist(y_forward, bins=100, label="y forward")
    axs[1][0].hist(x_backward, bins=100, label="x backward")
    axs[1][1].hist(y_backward, bins=100, label="y backward")

    for ax in axs.flat:
        ax.legend()

    plt.show()

    plot_min = -5
    plot_max = 5
    plt.hist2d(
        x_backward,
        y_backward,
        bins=200,
        range=[[plot_min, plot_max], [plot_min, plot_max]],
    )
    plt.title("Backward diffusion")
    plt.xlim(plot_min, plot_max)
    plt.ylim(plot_min, plot_max)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()

    plot_min = -10
    plot_max = 10
    plt.hist2d(
        x_forward,
        y_forward,
        bins=200,
        range=[[plot_min, plot_max], [plot_min, plot_max]],
    )
    plt.title("Forward diffusion")
    plt.xlim(plot_min, plot_max)
    plt.ylim(plot_min, plot_max)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()


if __name__ == "__main__":
    diffused_photons = diffusion(N_PHOTONS)

    no_diffusion = []
    back_diffusion = []
    forward_diffusion = []

    for photon in diffused_photons:
        if photon.current_diffusion == 0:
            no_diffusion.append(photon)
        elif photon.z <= 0:
            back_diffusion.append(photon)
        elif photon.z >= L:
            forward_diffusion.append(photon)
        else:
            raise ValueError("Photon did not leave the slab")

    total = len(diffused_photons)

    print(f"No diffusion : {len(no_diffusion) / total:.3f}")
    print(f"Beer Lambert : {np.exp(-alpha * L):.3f}")
    print(f"Forward diffusion : {len(forward_diffusion) / total:.3f}")
    print(f"Back diffusion : {len(back_diffusion) / total:.3f}")

    plot_results(forward_diffusion, back_diffusion)
