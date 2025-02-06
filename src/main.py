from scipy.ndimage import label
from phantom import EllipsoidPhantom
from propogation import Wave, project_wave_phantom, free_space_propagate, detect_wave

import matplotlib.pyplot as plt
from matplotlib_scalebar import scalebar
import numpy as np
from sys import exit

from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import matplotlib.font_manager as fm

E = 15e3 #eV
R = 0.005*5#0.08 #0.1  # propagation distance [m]
alpha = 1e-12
eta = 1e-8

Nx = 300   # detector grid shape
Ny = 300
upx = 1     # upsample multiple
sx = 1e-6   # pixel size [m]
sy = 1e-6
blur_fwhm = 1e-6

Nx_upx = Nx * upx
Ny_upx = Ny * upx
sx_upx = sx / upx
sy_upx = sy / upx

def born_preprocess(intensity, in_wave):
    return 0.5*(np.abs(intensity)/in_wave - 1)

def born_rytov_kernel(in_wave, gamma, R):
    kx, ky = in_wave.fftfreqs()
    k_squared = kx**2 + ky**2
    t1 = gamma*np.cos(np.pi * in_wave.wavelen*R*k_squared)
    t2 = np.sin(np.pi*in_wave.wavelen*R*k_squared)
    return t1+t2

def born_rytov_weiner(in_wave, gamma, R):
    kx, ky = in_wave.fftfreqs()
    k_squared = kx**2 + ky**2
    t1 = gamma*np.cos(np.pi * in_wave.wavelen*R*k_squared)
    t2 = np.sin(np.pi*in_wave.wavelen*R*k_squared)
    h = t1 + t2
    return np.conj(h)/(h*np.conj(h) + eta)


def paganin_kernel(in_wave, mu, delta, R):
    kx, ky = in_wave.fftfreqs()
    k_squared = kx**2 + ky**2
    kernel = (4*(np.pi**2)*R*(delta/mu)*k_squared + 1)
    return kernel

def apply_kernel_paganin(kernel, img, mu):
    img_fft = np.fft.fftshift(np.fft.fft2(img))
    filtered = np.abs(np.fft.ifft2(np.fft.ifftshift(img_fft*kernel)))
    return filtered

def fourier_product(kernel, img):
    img_fft = np.fft.fftshift(np.fft.fft2(img))
    return img_fft*kernel

def rytov_preprocess(detected, in_intensity):
    return 0.5*(np.log(detected/in_intensity))



phantom = EllipsoidPhantom(Tmax=50e-6, 
                        rx=30e-6, ry=40e-6, 
                        x0=5e-6,
                        y0=0, 
                        material_id='water',
                        material_composition='H(11.2)O(88.8)', 
                        density=1,
                        plane_shape=[Nx_upx, Ny_upx], 
                        pixel_scale=[sx_upx, sy_upx])

in_wave = Wave(amplitude=1.0,
            energy=E,    
            plane_shape=[Nx_upx, Ny_upx],
            pixel_size=[sx_upx, sy_upx])


#
# plt.imshow(phantom.projected_thickness())
# plt.colorbar()
# plt.show()

projection = project_wave_phantom(in_wave, phantom)
propogation = free_space_propagate(in_wave, projection, R)
detected = detect_wave(np.abs(propogation))

# print(np.min(detected))
plt.imshow(detected)
plt.colorbar()
plt.show()
#
# kernel = 1/paganin_kernel(in_wave, phantom.get_mu(E), phantom.get_delta_beta(E)[0], R)
# filtered = apply_kernel_paganin(kernel, detected/in_wave.I0, phantom.get_mu(E))
# thickness = -1/phantom.get_mu(E)*np.log(filtered)
#
#
# plt.imshow(np.real(thickness))
# plt.colorbar()
# plt.title("Paganin")
# #plt.savefig("paganin_simgle_dist.png")
# plt.show()
#
#
# #new - paganin
#
# H = paganin_kernel(in_wave, phantom.get_mu(E), phantom.get_delta_beta(E)[0], R)
# H_sq = H**2
# F = fourier_product(H, detected/in_wave.I0)
# log_arg = np.abs(np.fft.ifft2(np.fft.ifftshift(F/H_sq)))
# phase = (-1/2)*(phantom.get_delta()/ phantom.get_beta())*np.log(log_arg)
# thickness = phase*(in_wave.wavelen/(2*np.pi*phantom.get_delta()))
#
# plt.imshow(np.real(thickness))
# plt.colorbar()
# plt.title("Paganin - new")
# #plt.savefig("paganin_single_dist_new.png")
# plt.show()
# #Single distance Born with weiner
# kernel = born_rytov_weiner(in_wave, phantom.get_gamma(), R)
# filtered = fourier_product(kernel, born_preprocess(detected, in_wave.I0))
# phase = np.abs(np.fft.ifft2(np.fft.ifftshift(filtered)))
#
# plt.imshow(phase*(in_wave.wavelen/(2*np.pi*phantom.get_delta())))
# plt.colorbar()
# plt.title("Born + Weiner")
# plt.savefig("born_single_dist.png")
# plt.show()
#
# #Single distance Rytov with weiner
# kernel = born_rytov_weiner(in_wave, phantom.get_gamma(), R)
# filtered = fourier_product(kernel, rytov_preprocess(detected, in_wave.I0))
# phase = np.abs(np.fft.ifft2(np.fft.ifftshift(filtered)))
#
# plt.imshow(phase*(in_wave.wavelen/(2*np.pi*phantom.get_delta())))
# plt.colorbar()
# plt.title("Rytov + Weiner")
# plt.savefig("rytov_single_dist.png")
# plt.show()
#

R4 = [0.005, 0.008, 0.01, 0.012]
R4 = [r*5 for r in R4]
#R4 = [0.005*5]*4
R1 = R4[:1]
R2 = R4[:2]
R3 = R4[:3]
I = [free_space_propagate(in_wave, projection, r) for r in R4]
I = [detect_wave(np.abs(i)) for i in I]

I1 = I[:1]
I2 = I[:2]
I3 = I[:3]
I4 = I
#
# detected = np.mean(I4, axis=0)
# kernel = 1/paganin_kernel(in_wave, phantom.get_mu(E), phantom.get_delta_beta(E)[0], R)
# filtered = apply_kernel_paganin(kernel, detected/in_wave.I0, phantom.get_mu(E))
# thickness = -1/phantom.get_mu(E)*np.log(filtered)
#
#
# plt.imshow(np.real(thickness))
# plt.colorbar()
# plt.title("Paganin Avg")
# plt.savefig("paganin_samedist_avg.png")
# plt.show()

# kernel = born_rytov_weiner(in_wave, phantom.get_gamma(), R)
# filtered = fourier_product(kernel, born_preprocess(detected, in_wave.I0))
# phase = np.abs(np.fft.ifft2(np.fft.ifftshift(filtered)))
#
# plt.imshow(phase*(in_wave.wavelen/(2*np.pi*phantom.get_delta())))
# plt.colorbar()
# plt.title("Born + Weiner Avg")
# plt.savefig("born_samedist_avg.png")
# plt.show()
#
#
# kernel = born_rytov_weiner(in_wave, phantom.get_gamma(), R)
# filtered = fourier_product(kernel, rytov_preprocess(detected, in_wave.I0))
# phase = np.abs(np.fft.ifft2(np.fft.ifftshift(filtered)))
#
# plt.imshow(phase*(in_wave.wavelen/(2*np.pi*phantom.get_delta())))
# plt.colorbar()
# plt.title("Rytov + Weiner Avg")
# plt.savefig("rytov_samedist_avg.png")
# plt.show()

fig, axes = plt.subplots(2, 2, figsize=(10, 8), sharex= True, sharey=True)
fig.text(0.5, 0.04, 'pixels', ha='center')
fig.text(0.04, 0.5, 'pixels', va='center', rotation='vertical')
axes = axes.flatten()
fig_2, axes2 = plt.subplots(1,1)
tru = phantom.projected_thickness()
#
truth = phantom.projected_thickness()[110:190, 123:185]
# plt.imshow(truth)
# plt.show()
truvo = np.sum(truth*1e6)
print(f"truth: {truvo}")
j= 0
for I, R in zip([I1, I2, I3, I4], [R1, R2, R3, R4]):
    ax = axes[j]

    scalebar = AnchoredSizeBar(
        ax.transData,           # Transformation for the bar
        size=40,                # Size of the bar in data units (microns)
        label='40 µm',          # Label text
        loc='lower left',       # Location in the plot
        pad=0.5,                # Padding between bar and label
        color='white',          # Bar color
        frameon=False,          # No surrounding frame
        size_vertical=1,        # Thickness of the bar
        fontproperties=fm.FontProperties(size=10)  # Font size for the label
    )
    ax.add_artist(scalebar)

    H = [paganin_kernel(in_wave, phantom.get_mu(E), phantom.get_delta_beta(E)[0],r) for r in R]
    H_sq = [h**2 for h in H]
    F = [fourier_product(h, i/in_wave.I0) for h,i in zip(H,I)]
    num = np.sum(np.array(F), axis = 0)/len(R)
    den = (np.sum(np.array(H_sq), axis = 0))/len(R) + alpha
    log_arg = np.abs(np.fft.ifft2(np.fft.ifftshift(num/den)))
    phase = (-1/2)*(phantom.get_delta()/ phantom.get_beta())*np.log(log_arg)
    thickness = phase*(in_wave.wavelen/(2*np.pi*phantom.get_delta()))
    cax = ax.imshow(thickness, vmin = -5e-6, vmax= 4.8e-5)
    ax.set_title(f'Paganin - {j+1} image')

    j += 1



# Born Approximation
fig, axes = plt.subplots(2, 2, figsize=(10, 8), sharex= True, sharey=True)
fig.text(0.5, 0.04, 'pixels', ha='center')
fig.text(0.04, 0.5, 'pixels', va='center', rotation='vertical')
axes = axes.flatten()

j = 0
for I, R in zip([I1, I2, I3, I4], [R1, R2, R3, R4]):
    ax = axes[j]
    scalebar = AnchoredSizeBar(
        ax.transData,           # Transformation for the bar
        size=40,                # Size of the bar in data units
        label='40 µm',          # Label text
        loc='lower left',       # Location in the plot
        pad=0.5,                # Padding between bar and label
        color='white',          # Bar color
        frameon=False,          # No surrounding frame
        size_vertical=1,        # Thickness of the bar
        fontproperties=fm.FontProperties(size=10)  # Font size for the label
    )
    ax.add_artist(scalebar)

    H = [born_rytov_kernel(in_wave, phantom.get_gamma(),r) for r in R]
    H_sq = [h**2 for h in H]
    F = [fourier_product(h, born_preprocess(i, in_wave.I0)) for h,i in zip(H,I)]
    num = np.sum(np.array(F), axis = 0)/len(R)
    den = (np.sum(np.array(H_sq), axis = 0))/len(R) + alpha
    phase = np.abs(np.fft.ifft2(np.fft.ifftshift(num/den)))
    thickness = phase*(in_wave.wavelen/(2*np.pi*phantom.get_delta()))
    cax = ax.imshow(thickness, vmin =  -5e-6, vmax= 4.8e-5)
    ax.set_title(f'Born - {j+1} image')
    j += 1


#fig.colorbar(cax, ax=axes, orientation='vertical', fraction=0.02, pad=0.04)
#fig.savefig("born_same_labs.png")



# Rytov Approximation
fig, axes = plt.subplots(2, 2, figsize=(10, 8))
axes = axes.flatten()
j = 0
for I, R in zip([I1, I2, I3, I4], [R1, R2, R3, R4]):
    ax = axes[j]
    H = [born_rytov_kernel(in_wave, phantom.get_gamma(),r) for r in R]
    H_sq = [h**2 for h in H]
    F = [fourier_product(h, rytov_preprocess(i, in_wave.I0)) for h,i in zip(H,I)]
    num = np.sum(np.array(F), axis = 0)/len(R)
    den = (np.sum(np.array(H_sq), axis = 0))/len(R) + alpha
    phase = np.abs(np.fft.ifft2(np.fft.ifftshift(num/den)))
    thickness = phase*(in_wave.wavelen/(2*np.pi*phantom.get_delta()))
    cax = ax.imshow(thickness, vmin =  -5e-6, vmax= 4.8e-5)
    ax.set_title(f'Rytov - {j+1} distance')
    j += 1
        

# fig.savefig("rytov_multi.png")
# fig.show()
# axes2.legend()
# axes2.set_xlabel(r'$\mu$m')  # X-axis with micrometer
# axes2.set_ylabel(r'Thickness ($\mu$m)')
# fig_2.savefig("all_line_profiles_2.png")
#


fig.savefig("rytov.png")

