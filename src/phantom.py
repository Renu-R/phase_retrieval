import numpy as np
from consts import h,c,e

def get_delta_beta(energy, matcomp, density):
    return 1.043e-6, 3.553e-10

class Phantom():
    """
    A class to contain geometric and material properties of a voxelized, 
    computational phantom.    
    
    TODO: material properties, projection methods
    """
    def __init__(self, id, shape, voxel_size):
        self.name = id
        self.Nx, self.Ny, self.Nz = shape
        self.dx, self.dy, self.dz = voxel_size  # [m]
        self.FOVx = self.Nx*self.dx
        self.FOVy = self.Ny*self.dy
        self.FOVz = self.Nz*self.dz
        
    def grid_coordinates(self):  # Define the grid coordinates
        d_x_vals = np.linspace(-self.FOVx/2, self.FOVx/2, self.Nx) + self.dx/2 
        d_y_vals = np.linspace(-self.FOVy/2, self.FOVy/2, self.Ny) + self.dy/2
        d_X, d_Y = np.meshgrid(d_x_vals, d_y_vals)
        return d_X, d_Y

class EllipsoidPhantom(Phantom):
    def __init__(self, Tmax, rx, ry, x0, y0, material_id, material_composition, density,plane_shape, pixel_scale):
        if len(plane_shape) == 2:
            plane_shape = list(plane_shape) + [0]
            pixel_scale = list(pixel_scale) + [0]
        super().__init__(f'ellipsoid_{material_id}_{Tmax*1e6}um', plane_shape, pixel_scale)
        self.Tmax = Tmax
        self.rx = rx
        self.ry = ry
        self.x0 = x0
        self.y0 = y0
        self.matcomp = material_composition
        self.density = density
        self.d_thickness = None
        
    def projected_thickness(self):
        if self.d_thickness is None:
            d_X, d_Y = self.grid_coordinates()
            Z2 = 1 - ((d_X-self.x0)/self.rx)**2 - ((d_Y-self.y0)/self.ry)**2
            Z2[Z2 < 0] = 0
            d_thickness = self.Tmax * np.sqrt(Z2)
            self.d_thickness = d_thickness
        return self.d_thickness

    def get_delta(self):
        return 1.043e-6

    def get_beta(self):
        return 3.553e-10

    def get_gamma(self):
        return self.get_beta()/self.get_delta()
    
    def get_delta_beta(self, energy):
        return 1.043e-6, 3.553e-10

    def get_mu(self, energy):
        beta = self.get_delta_beta(None)[1]
        wavelength = h*c/(energy*e)
        return 4*np.pi*beta/wavelength
     
