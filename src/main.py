from phantom import EllipsoidPhantom
from propogation import Wave, project_wave_phantom, free_space_propagate, detect_wave

import matplotlib.pyplot as plt
import numpy as np


E = 15e3 #eV
R = 0.05 #0.1  # propagation distance [m]
blur_fwhm = 1e-6

Nx = 131   # detector grid shape
Ny = 131
upx = 1     # upsample multiple
sx = 0.8e-6   # pixel size [m]
sy = 0.8e-6

Nx_upx = Nx * upx
Ny_upx = Ny * upx
sx_upx = sx / upx
sy_upx = sy / upx

def paganin_kernel(in_wave, mu, delta, R):
    kx, ky = in_wave.fftfreqs()
    k_squared = kx**2 + ky**2
    kernel = 1/(4*(np.pi**2)*R*(delta/mu)*k_squared + 1)
    return kernel

def apply_kernel_paganin(kernel, img, mu):
    img_fft = np.fft.fftshift(np.fft.fft2(img))
    filtered = np.abs(np.fft.ifft2(np.fft.ifftshift(img_fft*kernel)))
    return -1/mu*np.log(filtered)



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

print(in_wave.wavenum)
print(phantom.get_mu(E))

#
# plt.imshow(phantom.projected_thickness())
# plt.colorbar()
# plt.show()

projection = project_wave_phantom(in_wave, phantom)
propogation = free_space_propagate(in_wave, projection, R)

#propogation = detect_wave(in_wave, propogation, upx, blur_fwhm)

plt.imshow(np.abs(propogation))
plt.colorbar()
plt.show()

kernel = paganin_kernel(in_wave, phantom.get_mu(E), phantom.get_delta_beta(E)[0], R)
thickness = apply_kernel_paganin(kernel, np.abs(propogation)/in_wave.I0, phantom.get_mu(E))

plt.imshow(np.real(thickness))
plt.colorbar()
plt.show()

