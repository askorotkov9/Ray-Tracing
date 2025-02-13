import seaborn as sns
import numpy as np
from typing import Union, Callable
from scipy.optimize import fsolve

from ray import *


material_nr=1.0,
def surface_apex(y, R, y0, x0):
    x = ((y-y0)**2)/(2*R) + x0  #parabolic profile
    return x


class Surface(object):
    def __init__(self, R, x_c, y0, lens_aperture, material_nr, record_rays=False, end_beam=False): #x0 - center position of the lens side on x-axis
        self.R = R
        self.x_c = x_c #center position of the lens on x-axis
        self.y0 = y0
        self.ray_bins = []
        self.record_rays = record_rays
        self.lens_aperture = lens_aperture
        #y axis shift
        self.y_min = - lens_aperture
        self.y_max = lens_aperture
        
        self.constant_nr = material_nr
        self.n_r = self._constant_nr
        
        self.end_beam = end_beam
        
    def _constant_nr(self, wavelength):
        return self.constant_nr

    def get_surface_apex(self, y):
        x = surface_apex(y, self.R, self.y0, self.x_c)
        return x    
    
    def get_surface_apex_prime(self, y, y0, R):
        return (y-y0)/R  #parabolic
    
    def get_surface_sidewall_prime(self, y, y0, R):
        return 0 #parabolic
    
    def ray_param_eq(self, t, x0, y0, theta, R):
        return ((y0 + t*np.sin(theta) - self.y0)**2)/(2*R) + self.x_c - (x0 + t*np.cos(theta))  #parabolic profile
    
    def ray_param_eq_prime(self, t, R, x0, y0, theta):
        return (np.sin(theta)*(y0 + t*np.sin(theta)-self.y0))/R - np.cos(theta) #parabolic profile
    
    def add_rays_into_bins(self, x, y, intensity, wavelength):
        self.ray_bins.append((x, y, intensity, wavelength))
    
    def get_tangent_vec(self, yp, y0, normalize=True):
        xp_p = self.get_surface_apex_prime(yp, y0, self.R)
        tangent_vec = np.empty(2)
        tangent_vec[0] = xp_p
        tangent_vec[1] = 1
        
        if normalize:
            tangent_vec = tangent_vec/norm(tangent_vec[0], tangent_vec[1])
        
        return tangent_vec
    
    def get_norm_vec(self, yp):
        tangent_vec = self.get_tangent_vec(yp, self.y0, normalize=True)
        normal_vec = np.empty(2)
        normal_vec[0] = -tangent_vec[1]
        normal_vec[1] = tangent_vec[0]
        
        return normal_vec
    
    def get_refraction(self, yp, ray: Ray, prev_n: float) -> np.array:
        
        n_r = self.n_r(ray.wavelength)
        norm_vec = -self.get_norm_vec(yp)
        #Calculate cosine from Snell's law
        cos_I = norm_vec[0] * ray.k[0] + norm_vec[1] * ray.k[1]
        sin_I = np.sqrt(1 - np.power(cos_I, 2))
        sin_Ip = prev_n * sin_I / n_r
        cos_Ip = np.sqrt(1 - np.power(sin_Ip, 2))
        #Calculate the refractive vector
        r_vec = ray.k
        nprpn = n_r * cos_Ip - prev_n * cos_I
        next_r = 1/n_r * (prev_n*r_vec+nprpn*norm_vec)
        
        return next_r
    
    def intersection(self, ray: Ray, prev_n: Union[Callable[[float], float], None], t_min, t_max):
        def unity_func(x):
            return 1.0
        if prev_n is None:
            prev_n = unity_func
                
        #Make a initial guess of t for newton-raphson solver
        # the t could be either at the edge of a surface or the center of a surface
        # depending on its shape
        
        t_min_p_1 = ray.estimate_t(self.get_surface_apex(self.lens_aperture))
        
        #apex
        t_end = fsolve(func=self.ray_param_eq, x0 = t_min_p_1, args=(ray.x_0, ray.y_0, ray.theta, self.R))
        x_end, y_end = ray.get_xy(t_end)

        # We have to check the point of intersection is within the boundary
        if (y_end <= self.y_max) and (y_end >= self.y_min):
            if self.record_rays:
                self.add_rays_into_bins(x_end, y_end, ray.intensity, ray.wavelength)
            next_r = self.get_refraction(y_end, ray, prev_n = prev_n(ray.wavelength))
            new_theta = np.arctan2(next_r[1], next_r[0])
            
            ray.update_after_intersection(t_end, new_theta = new_theta, end_beam = self.end_beam)
            
    def render(self, ax):
        rs = np.linspace(-self.lens_aperture, self.lens_aperture, 1000)
        xs = self.get_surface_apex(rs)

        ax.plot(xs, rs, color='black', linewidth=1)
        ax.set_xlabel("x, m")
        ax.set_ylabel("y, m")
        sns.set_theme(style = 'whitegrid', font = 'calibri', font_scale = 1.3)