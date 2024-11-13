import numpy as np
from scipy.signal import convolve2d
from consts import h,c,e

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
    exit_wave : 2D cupy array
        The complex plane wave after it has been modulated through a phantom.
    prop_dist : float
        Distance from the exit_wave plane to the detector plane.

    Returns
    -------
    d_fsp_wave : 2D cupy array
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

def lorentzian2D(x, y, fwhm, normalize=True):
    """
    Generate a 2D Lorentzian kernel.
    x, y : 1D cupy array
        Grid coordinates [arbitrary length]
    fwhm : float
        Full-width at half-max of the Lorentzian (units must match x,y)
    """
    gamma = fwhm/2
    X, Y = np.meshgrid(x, y)
    kernel = gamma / (2 * np.pi * (X**2 + Y**2 + gamma**2)**1.5)
    if normalize:
        kernel = kernel / np.sum(kernel)
    return kernel

def block_mean_2d(arr, Nblock):
    """
    Computes the block mean over a 2D array `arr`
    in sublocks of size Nblock x Nblock.
    If an axis size of arr is not an integer multiple 
    of Nblock, then zero padding is added at the end.

    This will cause image artifacts when fringes are too close
    relative to pixel size.
    """
    Ny, Nx = arr.shape
    padx, pady = 0, 0
    if Nx%Nblock != 0:
        padx = Nblock - Nx%Nblock
    if Ny%Nblock != 0:
        pady = Nblock - Ny%Nblock
    d_arr = np.array(arr)
    d_arr_pad = np.pad(d_arr, [(0, pady), (0, padx)])
    Ny, Nx = d_arr_pad.shape
    d_block_mean = d_arr_pad.reshape(Ny//Nblock, Nblock, Nx//Nblock, Nblock).mean(axis=(1,-1))
    return d_block_mean

def detect_wave(in_wave, wave, upsample_multiple, blur_fwhm=0):
    """
    Detect a complex plane wave by downsampling the array to the detector thickness
    with a 2D block mean (integer multiples of the pixel width only!) and
    possibly applying a Lorentzian blur.
    
    TODO : add noise.
    
    Parameters
    ----------
    in_wave : Wave
        Contains information about the initial wave geometry.
    wave : 2D cupy array
        The complex plane wave incident on the detector.
    upsample_multiple : int
        Number of pixels over which to take the block mean for downsampling.
    blur_fwhm : float
        Full-width at half-max of the Lorentzian kernel, which defines the
        point-spread-function of the detector.
        If blur_fwhm == 0, no blurring is applied (ideal).

    Returns
    -------
    d_detected_wave : 2D cupy array
        Real detected plane wave.
    """
    print(blur_fwhm) 
    if blur_fwhm > 0:
        # choose a sufficiently large kernel
        N_kernel = int(2 * blur_fwhm / in_wave.dx) + 1
        if (N_kernel > 2*in_wave.Nx + 1):
            N_kernel = 2*in_wave.Nx + 1  # crop if too large
        if (N_kernel%2 == 0):
            N_kernel -= 1
        # apply blur
        FOV_kern_x, FOV_kern_y = N_kernel*in_wave.dx, N_kernel*in_wave.dy
        d_x = np.linspace(-FOV_kern_x/2, FOV_kern_x/2, N_kernel)
        d_y = np.linspace(-FOV_kern_y/2, FOV_kern_y/2, N_kernel)
        d_lorentzian2d = lorentzian2D(d_x, d_y, blur_fwhm)
        d_wave_blurred = convolve2d(np.abs(wave), d_lorentzian2d, mode='same')
        d_detected_wave = block_mean_2d(d_wave_blurred, upsample_multiple)
    else:
        d_detected_wave = block_mean_2d(np.abs(wave), upsample_multiple)
    return d_detected_wave
