import numpy as np
from scipy.signal import convolve2d
from consts import h,c,e
from scipy.ndimage import convolve

def energy_to_wavelen(energy):
    return h*c/(energy*e)

def energy_to_wavenum(energy):
    return 2*np.pi*e*energy/(h*c)

class Wave():
    """
    A class to handle an x-ray wave front. Currently only for a coherent,
    monochromatic plane wave.
    """
    def __init__(self, amplitude, energy, plane_shape, pixel_size):  # initialize the plane
        self.I0 = amplitude
        self.energy = energy  # [eV]
        self.Nx, self.Ny = plane_shape
        self.dx, self.dy = pixel_size  # [m]
        self.wavelen = energy_to_wavelen(energy)
        self.wavenum = energy_to_wavenum(energy)
        self.FOVx = self.Nx*self.dx
        self.FOVy = self.Ny*self.dy
        
    def fftfreqs(self):   # Compute centered, discrete frequency samplings on the grid for DFT
        d_fx_vals = np.fft.fftfreq(self.Nx, self.dx)  # Frequency in x-direction
        d_fy_vals = np.fft.fftfreq(self.Ny, self.dy)  # Frequency in y-direction
        d_fx_vals = np.fft.fftshift(d_fx_vals)  # Shift the x frequencies
        d_fy_vals = np.fft.fftshift(d_fy_vals)  # Shift the y frequencies
        
        d_fx_grid, d_fy_grid = np.meshgrid(d_fx_vals, d_fy_vals)
        return d_fx_grid, d_fy_grid
    
    def update_energy(self, new_energy):
        self.energy = new_energy
        self.wavelen = energy_to_wavelen(new_energy)
        self.wavenum = energy_to_wavenum(new_energy)

def project_wave_phantom(in_wave, phantom1):
    """
    Use the projection approximation to modulate a coherent, monochromatic 
    plane wave through a single analytical phantom. Currently, there are two 
    options for analytical phantoms: a sphere or a cylinder. This allows one 
    to simulate a simple 3D phantom without the need to generate a voxelized 
    phantom file.

    """
    
    # Get phantom parameters
    d_t1 = phantom1.projected_thickness()
    thickness = d_t1.max()  # Only one phantom, so we use its thickness 
    delta1, beta1 = phantom1.get_delta_beta(in_wave.energy)
    
    # Project input wave through the phantom
    proj_delta = d_t1 * delta1  
    proj_beta = d_t1 * beta1
    d_exit_wave = in_wave.I0 * np.exp(1j*in_wave.wavenum * ( proj_delta + 2*1j*proj_beta))* np.exp(1j * in_wave.wavenum * thickness)

    print(np.max(np.abs(d_exit_wave)))

    return d_exit_wave

def free_space_propagate(in_wave, exit_wave, R):
    """
    Compute wave after propagating through free space.
    Applies the Fresnel propagator in the Fourier domain.

    Parameters
    ----------
    in_wave : Wave
        Contains information about the incident wave geometry.
    exit_wave : 2D numpy array
        The complex plane wave after it has been modulated through a phantom.
    prop_dist : float
        Distance from the exit_wave plane to the detector plane.

    Returns
    -------
    d_fsp_wave : 2D numpy array
        Complex plane wave after FSP.
    
    """

    if R==0:
        return exit_wave
    k = in_wave.wavenum
    FX, FY = in_wave.fftfreqs()
    H = np.exp(1j*(2 * np.pi / in_wave.wavelen) * R) * np.exp(1j * np.pi * in_wave.wavelen * R * (FX**2 + FY**2))
    #H = np.exp(1j * k / (2 * R) * (FX**2 + FY**2))
    exit_wave_fft = np.fft.fftshift(np.fft.fft2(exit_wave))
    d_fsp_wave =np.fft.ifft2(np.fft.ifftshift(exit_wave_fft*H))
    return d_fsp_wave


def detect_wave_1(image: np.ndarray, noise_level: float = 5e-4) -> np.ndarray:
    """
    Simulates an X-ray image detector by adding Lorentzian noise to the input image.

    Parameters:
        image (np.ndarray): Input 2D array representing the image.
        noise_level (float): Scale of Lorentzian noise to add (relative to image intensity range).

    Returns:
        np.ndarray: The detected image with added noise.
    """
    if not isinstance(image, np.ndarray):
        raise TypeError("Input image must be a NumPy array.")

    if image.ndim != 2:
        raise ValueError("Input image must be a 2D array.")

    # Normalize the image to [0, 1] for consistent noise application
    image_min, image_max = image.min(), image.max()
    normalized_image = (image - image_min) / (image_max - image_min)

    # Add Lorentzian noise: Generate noise from the Cauchy distribution
    noise = np.random.standard_cauchy(normalized_image.shape) * noise_level
    noisy_image = normalized_image + noise

    # Clip values to ensure valid intensity range [0, 1]
    noisy_image = np.clip(noisy_image, 0, 1)

    # Scale back to the original intensity range
    detected_image = noisy_image * (image_max - image_min) + image_min

    return detected_image

def lorentzian_kernel(fwhm: float, size: int = 3) -> np.ndarray:
    x = np.linspace(-size // 2, size // 2, size)
    y = np.linspace(-size // 2, size // 2, size)
    x, y = np.meshgrid(x, y)
    r = np.sqrt(x**2 + y**2)
    gamma = fwhm / 2
    kernel = 1 / (1 + (r / gamma)**2)
    kernel /= np.sum(kernel)
    return kernel

def detect_wave(image: np.ndarray, blur: float = 1e-6, noise: float = 10e-4) -> np.ndarray:
    """
    Simulates an X-ray image detector by adding Lorentzian noise to the input image.
    
    Parameters:
        image (np.ndarray): Input 2D array representing the image.
        pixel_size (float): Size of the pixels (in some units, e.g., microns or millimeters).
        fwhm (float): Full Width at Half Maximum of the Lorentzian noise in the same units as pixel_size.
        
    Returns:
        np.ndarray: The detected image with added Lorentzian noise.
    """
    if not isinstance(image, np.ndarray):
        raise TypeError("Input image must be a NumPy array.")

    if image.ndim != 2:
        raise ValueError("Input image must be a 2D array.")

    kernel = lorentzian_kernel(blur)
    blurred_image = convolve2d(image, kernel, mode='same', boundary='wrap')
    #blurred_image = image

    # Calculate the scale parameter (gamma) from the FWHM
    gamma = noise / 2
    
    # Normalize the image to [0, 1] for consistent noise application
    image_min, image_max = image.min(), image.max()
    normalized_image = (blurred_image - image_min) / (image_max - image_min)

    # Add Lorentzian noise: Generate noise from the Cauchy distribution
    noise = np.random.standard_cauchy(normalized_image.shape) * gamma
    noisy_image = normalized_image + noise

    # Clip values to ensure valid intensity range [0, 1]
    noisy_image = np.clip(noisy_image, 0, 1)

    # Scale back to the original intensity range
    detected_image = noisy_image * (image_max - image_min) + image_min

    return detected_image
