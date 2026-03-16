import numpy as np
from matplotlib import pyplot as plt





#Constants
alpha = 0.5 #cm-1
L = 6 #cm



#Questions

#1
"""
If a Z is a random variable following an exponential pdf

Z(z) = lambda*exp(-lambda*z)

The CDF (Fonction de Repartition) is then given by 

FZ(z) = integral_{-inf}^{z}z(u)du = lambda/(-lambda)[exp(-lambda*u)]_{-inf}^{z}
FZ(z) = (-1)*[exp(-lambda*z) - 1] = 1 - [exp(-lambda*z)

The sampling rate lambda can be taken as the linear coefficient loss alpha

Another way to see it

The transmited power to a position M is homogenous to the power that is transmitted to that position
over the total incident power

P(Z>M) = Transmitted/Incident = exp(-alpha*M) , then the pdf is found by derivating this

"""
#2
"""
FZ(z) = 1 - [exp(-lambda*z) -> ln(1 - FZ) = -lambda*z -> z = -ln(1 - FZ)/lambda = FZ^{-1}

By the same sampling law that we saw earlier in the Monte Carlo TP, tp2, we know that if a random variable Z follows
a certain pdf , then a sample of Z can be found as : 

Z_i = FZ^{-1}(U_i)

Where U_i is a sample from a uniform law with 0,1 hi lo

"""
#4
"""
We do the integral of the solid angle from theta from 0 to pi/2

Delta Omega = 2pi * integral_{0}^{theta}sin(t)dt = 2pi * [-cos(t)]_{0}^{theta} = 2pi * [1 - cos(theta)]

Donc

FT = (1/2) * [1 - cos(theta)]

cos(theta) = 1 - 2FT = FT^(-1)

"""
class Photon :
    def __init__(self , z , costheta , phi):
        self.trajectory = []
        self.z = z
        self.costheta = costheta
        self.phi = phi
        self.x , self.y = spherical_to_cartesian(z,costheta,phi)
        self.current_diffusion = None

def ZCDF(u):
    return 1 - np.exp(-alpha*u)

def inverse_ZCDF(u):
    return (-np.log(1 - u ))/alpha

def thetaCDF(u):
    return (1/2) * [1 - np.cos(u)]

def inverse_costhetaCDF(u):
    return 1 - 2*u

def spherical_to_cartesian(r,costheta,phi):
    theta = np.arccos(costheta)
    sintheta = np.sin(theta)

    x = r*sintheta*np.cos(phi)
    y = r*sintheta*np.sin(phi)

    return x,y



def diffusion(n_photons):
    photon_list = []
    photons_left = n_photons
    current_difusion = 0
    in_photons_z = np.zeros(shape=(n_photons,))
    in_photons_costheta = np.ones(shape=n_photons,)
    in_photons_phi = np.zeros(shape=(n_photons,))

    while photons_left > 0 :
        r_uniform_samples = np.random.uniform(low = 0 , high=1 , size = photons_left)
        r_samples = inverse_ZCDF(r_uniform_samples)

        if current_difusion == 0:
            costheta_samples = np.ones_like(r_samples)
        else:
            costheta_uniform_samples = np.random.uniform(low = 0 , high = 1 , size = photons_left )
            costheta_samples = inverse_costhetaCDF(costheta_uniform_samples)

        phi_samples = np.random.uniform(low=0 , high=2*np.pi , size = photons_left)
        
        z_positions = np.multiply(r_samples,costheta_samples)

        z_positions = z_positions + in_photons_z

        mask_z_is_out = (z_positions < 0) | (z_positions > L)
        if sum(mask_z_is_out) != 0:
            old_x_positions , old_y_positions = spherical_to_cartesian(in_photons_z[mask_z_is_out] , 
                                                                       in_photons_costheta[mask_z_is_out] , 
                                                                       in_photons_phi[mask_z_is_out])
            out_photons = np.vstack([z_positions[mask_z_is_out] , 
                                    costheta_samples[mask_z_is_out],
                                    phi_samples[mask_z_is_out]]).T
            
            curr_list = list(map(lambda x: Photon(x[0] , x[1] , x[2]),out_photons))
            for i in range(len(curr_list)):
                curr_list[i].current_diffusion = current_difusion
                curr_list[i].x += old_x_positions[i]
                curr_list[i].y += old_y_positions[i]

            photon_list.extend(curr_list)

        in_photons_z = z_positions[~mask_z_is_out]
        in_photons_costheta = costheta_samples[~mask_z_is_out]
        in_photons_phi = phi_samples[~mask_z_is_out]


        photons_left = n_photons - len(photon_list)
        current_difusion = current_difusion + 1


    return photon_list

        

    



diffused_photons = diffusion(100000)

no_diffusion = []
back_diffusion = []
forward_diffusion = []
x_positions = []
y_positions = []

x_positions_forward = []
y_positions_forward = []

x_positions_backward = []
y_positions_backward = []



for p in diffused_photons:
    x_positions.append(p.x)
    y_positions.append(p.y)
    if p.current_diffusion == 0:
        no_diffusion.append(p)
    else:
        if p.z <= 0 :
            back_diffusion.append(p)
            x_positions_backward.append(p.x)
            y_positions_backward.append(p.y)
        elif p.z >= L:
            forward_diffusion.append(p)
            x_positions_forward.append(p.x)
            y_positions_forward.append(p.y)
        else:
            raise ValueError("forcas malignas")

print(f"No diffusion : {len(no_diffusion)/len(diffused_photons):.3f}")
print(f"Beer Lambert : {np.exp(-3):.3f}")
print(f"Forward diffusion : {len(forward_diffusion)/len(diffused_photons):.3f}")
print(f"Back diffusion : {len(back_diffusion)/len(diffused_photons):.3f}")

fig , axs = plt.subplots(2,2)

axs[0][0].hist(x_positions_forward , bins = 100 , label = "x forward")
axs[0][1].hist(y_positions_forward , bins = 100 , label = "y forward")

axs[1][0].hist(x_positions_backward , bins = 100 , label = "x backwards")
axs[1][1].hist(y_positions_backward , bins = 100 , label = "y backwards")

for ax in axs.flat:
    ax.legend()
plt.show()

min = -5
max = 5

plt.hist2d(x_positions_backward, y_positions_backward, bins=200, range=[[min, max], [min, max]])
plt.title("backwards difusion")
plt.xlim(min, max)
plt.ylim(min, max)
plt.xlabel("x")
plt.ylabel("y")
plt.show()

min = -10
max = 10
plt.hist2d(x_positions_forward, y_positions_forward, bins=200, range=[[min, max], [min, max]])
plt.title("forward difusion")
plt.xlim(min, max)
plt.ylim(min, max)
plt.xlabel("x")
plt.ylabel("y")
plt.show()



