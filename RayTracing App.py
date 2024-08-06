import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.ticker import EngFormatter
from typing import Union, Callable
from scipy.optimize import fsolve
import time
from matplotlib.widgets import Cursor

import math


#Source set
d = 30*1e-6 #space between apexes, um
x_c = 0 #center point of the first lens
y0 = 0 #beam offset in y-axis
E = 9.25 #float(input("Enter in '12.0' format, from 1 to 60 keV (step 0.5): ")) #input()
source_size = 11.727*1e-6
divergence = 28.3*1e-5
print("Energy, kev: ", E, "\nSource size, um: ", source_size, "\nDivergence, um:", divergence)

#Determines the quality of the plot
point_num=120

# Lens set, all units in um:
R500 = {'radius': 500*1e-6, 'lens aperture': 1393*1e-6}
R200 = {'radius': 200*1e-6, 'lens aperture': 800*1e-6}
R100 = {'radius': 100*1e-6, 'lens aperture': 600*1e-6}
R50 = {'radius': 50*1e-6, 'lens aperture': 440*1e-6}
N1 = 38 #input("Enter numer of lenses")
N2 = 63 
dist_crl = 1.775 *1E-3 #distance between lenses

#output index of refraction and total linear coefficient from csv
dataframe = pd.read_csv('C:/Users/askor/OneDrive/Рабочий стол/Xoptics/2024/RayTracing App/materials/Be 2keV-100keV.csv', sep = ';')

print(dataframe.head())
index_n = dataframe[dataframe.isin([E]).any(axis=1)] #searching for the index of refraction by the input energy
delta = 3.976746331307E-6   #8.5004135933024E-7 #float(index_n['Delta']) #searching for the delta by the input energy
material_n = 1-delta	#float(index_n['n'])
wavelength_a = float(index_n['Wavelength, A']) #searching for the wavelength by the input energy 
#beta = float(index_n['Beta']) #searching for the delta by the input energy
total_linear_coeff = float(index_n['Linear attenuation coefficient (photoelectric part), 1/um']) #searching for the total linear coefficient by the input energy
print("n =", material_n, "\nwavelength = ", wavelength_a, ' A' "\ndelta = ", delta, "\nmu = ", total_linear_coeff)

#Ray part
f_point = []
y_point_for_plt = []
Aeff = []
Res = []
numerical_aperture = []
depth_of_field = []
material_nr = 1.0 

def _init_vec(x,y):
    vec=np.empty(2)
    vec[0]=x
    vec[1]=y
    return vec

def norm(x, y):
    return math.sqrt(math.pow(x,2)+math.pow(y,2))

class Ray(object):
    """
    The ray is a set of line segments. Each segment is represented by a vector v=[x,y]
    Ray-tracing problem also requires parameterized expressions, i.e.,
    x(t)=x_0+cos(theta)*t
    y(t)=y_0+sin(theta)*t

    x_0, y_0, theta are updated after an intersection with a surface occurs.
    Lists are used to store the old values of x_0, y_0 and theta so that a figure renderer can plot all the segments


    The ray data structure is:

        |                         |                                 |
        |                         |                                 |
        |                         |                                 |
        |                         |                                 |
    starting point              surface 1                          surface 2

       t_0                      t_1=                              t_2=
                                t_end+t_0                         t_end+t_1

     self.path: [v0,theta_0]      [[v_0,theta_0],[v_1,theta_1]]   [[v_0,theta_0],[v_1,theta_1],[v_2,theta_2]]
     self.end_ts: [Inf]           [t_1,Inf]                       [t_1,t_2,Inf]
     theta_x is raw angle from surface X, x=0 is starting point.
    v_0 are the turning points of the ray vector. Every time a ray is intersected with a surface, the value of v_0 is updated
    t_end is the ending "time" t of each line segment

    """

    def __init__(self, x0, y0, theta, wavelength=600):
        """
        Initialize a beam

        :param x_0: starting x position
        :param y_0: starting y position
        :param theta: traveling angle (theta)
        """
        self.x_0, self.y_0 = x0, y0
        self.theta = theta
        self.dt = 0
        self.v_0 = _init_vec(self.x_0, self.y_0)  # vector that stores the initial point of the ray
        self.paths = [[np.copy(self.v_0), np.copy(self.theta)]]
        self.I_0 = 100
        self.I_i = []
        self.end_ts = [float('Inf')]
        self.k = _init_vec(np.cos(self.theta), np.sin(self.theta))
        self.intensity = 1
        self.wavelength = wavelength
        self.x_points = []

    def update_after_intersection(self, t_end, new_theta, end_beam=False):
        """
        Update the state of a ray, including:
        - a new starting point: [z_0,y_0]
        - angle of directional cosine: theta

        :param t_end: point of intersection with the surface
        :param new_theta: new traveling angle theta (radians)
        :param total_linear_coeff: attenuation coefficient, assigned by the Surface object
        :param end_beam: True if the ray will be stopped at this surface
        :return: None
        """

        self.v_0 += self.k * t_end

        self.x_0 = self.v_0[0]
        self.y_0 = self.v_0[1]
        
        self.update_theta(new_theta)
        next_t = t_end + self.dt
        self.dt = next_t
        self.end_ts[-1] = next_t

        self.paths.append([np.copy(self.v_0), np.copy(self.theta)])
        
        #self.attenu_x.append(self.x_0)
        #self.attenu_y.append(self.y_0)

        #
        # self.d_intensity = ((np.abs(self.attenu_x[-1]) - np.abs(self.attenu_x[-2]))**2 + (np.abs(self.attenu_y[-1]) - np.abs(self.attenu_y[-2])))**(1/2) 
        

        #if material_nr == 1.0:
        #    self.d_intensity.append(((self.attenu_x[-1] - self.attenu_x[-2])**2 + (self.attenu_y[-1] - self.attenu_y[-2]))**(1/2))
        #    self.I_i = self.I_0*np.exp(-total_linear_coeff*self.d_intensity*10**(-6))
        #    self.I_0 = self.I_i

        
        #self.I_i = self.I_0*np.exp(-total_linear_coeff*self.theta_test)
        #self.theta_test = np.arcsin((1/total_linear_coeff)*np.sin(self.theta_test))
        #self.I_0 = self.I_i

        if not end_beam:
            self.end_ts.append(float('Inf'))

    def get_xy(self, delta_t):
        """
        Get x and y positions from parameter t

        :param delta_t:
        :return: np.array [x,y]
        """
        vv = self.v_0 + self.k * delta_t
        return vv[0], vv[1]

    def estimate_t(self, xp:float):
        """
        Calculate t from a given x-position xp

        :param xp: a value of x
        :return: estimated t
        """

        t = (xp - self.v_0[0]) / self.k[0]

        return t

    def update_theta(self, new_theta):
        """
        Update the traveling angle theta

        :param new_theta: new traveling angle theta of the ray
        :return: None
        """
        self.theta = new_theta
        self.k = _init_vec(np.cos(self.theta), np.sin(self.theta))

    def render(self, ax: Axes, time_of_fights, color='C0'):
        """
        Render the ray start from the most recent reracted surface

        :param ax: Matplotlib Axes to plot on
        :param time_of_fights: the stopping time that the beam ends
        :param color: matplotlib color
        :return: None
        """
        v_e = self.v_0 + time_of_fights * self.k

        #Make divergent beam here

        v_for_plots = np.vstack((self.v_0, v_e))
        xs = v_for_plots[:, 0]
        ys = v_for_plots[:, 1]

        ax.plot(xs, ys, color=color)

    def get_k_from_theta(self, theta):

        k = _init_vec(np.cos(theta), np.sin(theta))

        return k
    
    def render_all(self, ax, time_of_flights, color=None):
        """
        Plot all rays on the axes.

        :param ax: axes to be plotted on
        :param time_of_flights: end travel time of the ray
        :param color: matplotlib color, such as 'C0', 'C1' or 'blue', 'red'. Set None for automatic colors.
        :return:
        """

        prev_t = 0
        for idx in range(len(self.end_ts)):
            v_0, theta = self.paths[idx]
            end_t = self.end_ts[idx]
            k = self.get_k_from_theta(theta)

            if time_of_flights > end_t:
                v_e = v_0 + (end_t - prev_t) * k
            else:
                v_e = v_0 + (time_of_flights - prev_t) * k
            v_for_plots = np.vstack((v_0, v_e))
            xs = v_for_plots[:, 0]
            ys = v_for_plots[:, 1]
            prev_t = end_t
            if color is None:
                plot_color = 'C{}'.format(idx)
            else:
                plot_color = sns.color_palette('dark')
            ax.plot(xs, ys, color=plot_color, linewidth=1.2)

# Old surface definition
material_nr=1.0,
def surface_apex(y, R, y0, x0):
    #x = (1/(2*R))*(y-y0)**2+(b/R)*(y-y0)+x0 #axicon
    #x = R/np.absolute(R)*(((d/2)**2+(d/2*(y-y0)**2)/np.absolute(R))**(1/2) + x0)  #hyperbola
    x = ((y-y0)**2)/(2*R) + x0   #parabolic
    return x

#nr = input() #показатель преломления ()
class Surface(object):
    def __init__(self, R, x_c, y0, lens_aperture, material_nr, record_rays=False, end_beam=False): #x0 - center position of the lens side on x-axis
        self.R = R
        self.x_c = x_c #center position of the lens on x-axis
        self.y0 = y0
        self.ray_bins = []
        self.record_rays = record_rays
        self.lens_aperture = lens_aperture
        #для сдвигу по y
        self.y_min = - lens_aperture
        self.y_max = lens_aperture
        
        self.constant_nr = material_nr
        self.n_r = self._constant_nr
        
        self.end_beam = end_beam
        
    def _constant_nr(self, wavelength):
        #self.nr = input() #для линзы
        return self.constant_nr

    def get_surface_apex(self, y):
        x = surface_apex(y, self.R, self.y0, self.x_c)
        return x    
    
    def get_surface_apex_prime(self, y, y0, R):
        #return (y-y0)/R + b/R #axicon
        #return -R/np.absolute(R)*(-(1/2*((d/2)**2 - (d/2*(y-y0)**2)/R)**(-1/2))*(d/R)*(y-y0)) #hyperbola
        return (y-y0)/R  #parabolic
    
    def get_surface_sidewall_prime(self, y, y0, R):
        #return (y-y0)/R + b/R #axicon
        #return -R/np.absolute(R)*(-(1/2*((d/2)**2 - (d/2*(y-y0)**2)/R)**(-1/2))*(d/R)*(y-y0)) #hyperbola
        return 0 #parabolic
    
    def ray_param_eq(self, t, x0, y0, theta, R):
        #return 1/(2*R)*((y0+t*np.sin(theta))**2-self.y0)**2+b/R*((y0+t*np.sin(theta))**2-self.y0)-(x0+t*np.cos(theta)-self.x_c) #axicon
        #return R/np.absolute(R)*((d/2)**2 + ((d/2)*(y0+t*np.sin(theta)-self.y0)**2)/np.absolute(R))**(-1/2) + self.x_c - (x0+t*np.cos(theta)) #hyperbola
        return ((y0 + t*np.sin(theta) - self.y0)**2)/(2*R) + self.x_c - (x0 + t*np.cos(theta))  #parabolic
    
    def ray_param_eq_prime(self, t, R, x0, y0, theta):
        #return (2*(y0+t*np.sin(theta))*np.sin(theta)*((y0+t*np.sin(theta))**2-self.y0)**2)/R+b/R(2*np.sin(theta)*(y0+t*np.sin(theta)))-np.cos(theta) #axicon
        #return -R/np.absolute(R)*(((d/2)**2 - d/(2*np.absolute(R))*(y0 + t*np.sin(theta) - self.y0)**2)**(-1/2)*(y0 + t*np.sin(theta) - self.y0)*np.sin(theta)) - np.cos(theta) #hyperbola
        return (np.sin(theta)*(y0 + t*np.sin(theta)-self.y0))/R - np.cos(theta) #parabolic
    
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
    
    #unity func можно убрать и вводить 1.0 для воздуха при инициализации
    
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


#Define a divergent bunch of rays
ray_number = 5
ray_number_divergence = 5
#divergence = np.linspace(-1E-6, 1E-6, num = ray_number)
y_start = np.linspace(-source_size/2, source_size/2, num = ray_number)
x_start = np.zeros_like(y_start)
theta = np.linspace(-divergence/2, divergence/2, num = ray_number_divergence)
rays = []
print("Number of point sources", y_start.shape[0], "\nNumber of rays from one point: ", theta.shape[0])

#TF1
#обозначать какие линзы в тф через переменную - здесь
asf1 = Surface(lens_aperture = R500['lens aperture']/2, material_nr = material_n, x_c = 27, y0 = y0, R = -R500['radius']) #воздух - линза
bsf1 = Surface(lens_aperture = R500['lens aperture']/2, material_nr = 1.0, x_c = 27 + d/2, y0 = y0, R = R500['radius']) #Be - воздух

asf2 = Surface(lens_aperture = R500['lens aperture']/2, material_nr = material_n, x_c= bsf1.get_surface_apex(R500['lens aperture']/2) + (bsf1.get_surface_apex(R500['lens aperture']/2) - bsf1.get_surface_apex(x_c)) + dist_crl, y0=y0, R=-R500['radius']) #воздух - линза
bsf2 = Surface(lens_aperture = R500['lens aperture']/2, material_nr = 1.0, x_c=asf2.get_surface_apex(x_c)+d/2, y0=y0, R=R500['radius']) #воздух - линза (Be)

asf3 = Surface(lens_aperture = R500['lens aperture']/2, material_nr = material_n, x_c= bsf2.get_surface_apex(R500['lens aperture']/2) + (bsf2.get_surface_apex(R500['lens aperture']/2) - bsf2.get_surface_apex(x_c)) + dist_crl, y0=y0, R=-R500['radius']) #воздух - линз
bsf3 = Surface(lens_aperture = R500['lens aperture']/2, material_nr = 1.0, x_c=asf3.get_surface_apex(x_c)+d/2, y0=y0, R=R500['radius']) #воздух - линза (Be)

asf4 = Surface(lens_aperture = R500['lens aperture']/2, material_nr = material_n, x_c= bsf3.get_surface_apex(R500['lens aperture']/2) + (bsf3.get_surface_apex(R500['lens aperture']/2) - bsf3.get_surface_apex(x_c)) + dist_crl, y0=y0, R=-R500['radius']) #воздух - линза
bsf4 = Surface(lens_aperture = R500['lens aperture']/2, material_nr = 1.0, x_c=asf4.get_surface_apex(x_c)+d/2, y0=y0, R=R500['radius']) #воздух - линза (Be)

asf5 = Surface(lens_aperture = R500['lens aperture']/2, material_nr = material_n, x_c= bsf4.get_surface_apex(R500['lens aperture']/2) + (bsf4.get_surface_apex(R500['lens aperture']/2) - bsf4.get_surface_apex(x_c)) + dist_crl, y0=y0, R=-R500['radius']) #воздух - линза
bsf5 = Surface(lens_aperture = R500['lens aperture']/2, material_nr = 1.0, x_c=asf5.get_surface_apex(x_c)+d/2, y0=y0, R=R500['radius']) #воздух - линза (Be)

asf6 = Surface(lens_aperture = R500['lens aperture']/2, material_nr = material_n, x_c= bsf5.get_surface_apex(R500['lens aperture']/2) + (bsf5.get_surface_apex(R500['lens aperture']/2) - bsf5.get_surface_apex(x_c)) + dist_crl, y0=y0, R=-R500['radius']) #воздух - линза
bsf6 = Surface(lens_aperture = R500['lens aperture']/2, material_nr = 1.0, x_c=asf6.get_surface_apex(x_c)+d/2, y0=y0, R=R500['radius']) #воздух - линза (Be)

asf7 = Surface(lens_aperture = R500['lens aperture']/2, material_nr = material_n, x_c= bsf6.get_surface_apex(R500['lens aperture']/2) + (bsf6.get_surface_apex(R500['lens aperture']/2) - bsf6.get_surface_apex(x_c)) + dist_crl, y0=y0, R=-R500['radius']) #воздух - линза
bsf7 = Surface(lens_aperture = R500['lens aperture']/2, material_nr = 1.0, x_c=asf7.get_surface_apex(x_c)+d/2, y0=y0, R=R500['radius']) #воздух - линза (Be)

asf8 = Surface(lens_aperture = R500['lens aperture']/2, material_nr = material_n, x_c= bsf7.get_surface_apex(R500['lens aperture']/2) + (bsf7.get_surface_apex(R500['lens aperture']/2) - bsf7.get_surface_apex(x_c)) + dist_crl, y0=y0, R=-R500['radius']) #воздух - линза
bsf8 = Surface(lens_aperture = R500['lens aperture']/2, material_nr = 1.0, x_c=asf8.get_surface_apex(x_c)+d/2, y0=y0, R=R500['radius']) #воздух - линза (Be)

asf9 = Surface(lens_aperture = R500['lens aperture']/2, material_nr = material_n, x_c= bsf8.get_surface_apex(R500['lens aperture']/2) + (bsf8.get_surface_apex(R500['lens aperture']/2) - bsf8.get_surface_apex(x_c)) + dist_crl, y0=y0, R=-R500['radius']) #воздух - линза
bsf9 = Surface(lens_aperture = R500['lens aperture']/2, material_nr = 1.0, x_c=asf9.get_surface_apex(x_c)+d/2, y0=y0, R=R500['radius']) #воздух - линза (Be)

asf10 = Surface(lens_aperture = R500['lens aperture']/2, material_nr = material_n, x_c= bsf9.get_surface_apex(R500['lens aperture']/2) + (bsf9.get_surface_apex(R500['lens aperture']/2) - bsf9.get_surface_apex(x_c)) + dist_crl, y0=y0, R=-R500['radius']) #воздух - линза
bsf10 = Surface(lens_aperture = R500['lens aperture']/2, material_nr = 1.0, x_c=asf10.get_surface_apex(x_c)+d/2, y0=y0, R=R500['radius']) #воздух - линза (Be)

asf11 = Surface(lens_aperture = R500['lens aperture']/2, material_nr = material_n, x_c= bsf10.get_surface_apex(R500['lens aperture']/2) + (bsf10.get_surface_apex(R500['lens aperture']/2) - bsf10.get_surface_apex(x_c)) + dist_crl, y0=y0, R=-R500['radius']) #воздух - линза
bsf11 = Surface(lens_aperture = R500['lens aperture']/2, material_nr = 1.0, x_c=asf11.get_surface_apex(x_c)+d/2, y0=y0, R=R500['radius']) #воздух - линза (Be)

asf12 = Surface(lens_aperture = R500['lens aperture']/2, material_nr = material_n, x_c= bsf11.get_surface_apex(R500['lens aperture']/2) + (bsf11.get_surface_apex(R500['lens aperture']/2) - bsf11.get_surface_apex(x_c)) + dist_crl, y0=y0, R=-R500['radius']) #воздух - линза
bsf12 = Surface(lens_aperture = R500['lens aperture']/2, material_nr = 1.0, x_c=asf12.get_surface_apex(x_c)+d/2, y0=y0, R=R500['radius']) #воздух - линза (Be)

asf13 = Surface(lens_aperture = R500['lens aperture']/2, material_nr = material_n, x_c= bsf12.get_surface_apex(R500['lens aperture']/2) + (bsf12.get_surface_apex(R500['lens aperture']/2) - bsf12.get_surface_apex(x_c)) + dist_crl, y0=y0, R=-R500['radius']) #воздух - линза
bsf13 = Surface(lens_aperture = R500['lens aperture']/2, material_nr = 1.0, x_c=asf13.get_surface_apex(x_c)+d/2, y0=y0, R=R500['radius']) #воздух - линза (Be)

asf14 = Surface(lens_aperture = R500['lens aperture']/2, material_nr = material_n, x_c= bsf13.get_surface_apex(R500['lens aperture']/2) + (bsf13.get_surface_apex(R500['lens aperture']/2) - bsf13.get_surface_apex(x_c)) + dist_crl, y0=y0, R=-R500['radius']) #воздух - линза
bsf14 = Surface(lens_aperture = R500['lens aperture']/2, material_nr = 1.0, x_c=asf14.get_surface_apex(x_c)+d/2, y0=y0, R=R500['radius']) #воздух - линза (Be)

asf15 = Surface(lens_aperture = R500['lens aperture']/2, material_nr = material_n, x_c= bsf14.get_surface_apex(R500['lens aperture']/2) + (bsf14.get_surface_apex(R500['lens aperture']/2) - bsf14.get_surface_apex(x_c)) + dist_crl, y0=y0, R=-R500['radius']) #воздух - линза
bsf15 = Surface(lens_aperture = R500['lens aperture']/2, material_nr = 1.0, x_c=asf15.get_surface_apex(x_c)+d/2, y0=y0, R=R500['radius']) #воздух - линза (Be)

asf16 = Surface(lens_aperture = R500['lens aperture']/2, material_nr = material_n, x_c= bsf15.get_surface_apex(R500['lens aperture']/2) + (bsf15.get_surface_apex(R500['lens aperture']/2) - bsf15.get_surface_apex(x_c)) + dist_crl, y0=y0, R=-R500['radius']) #воздух - линза
bsf16 = Surface(lens_aperture = R500['lens aperture']/2, material_nr = 1.0, x_c=asf16.get_surface_apex(x_c)+d/2, y0=y0, R=R500['radius']) #воздух - линза (Be)

asf17 = Surface(lens_aperture = R500['lens aperture']/2, material_nr = material_n, x_c= bsf16.get_surface_apex(R500['lens aperture']/2) + (bsf16.get_surface_apex(R500['lens aperture']/2) - bsf16.get_surface_apex(x_c)) + dist_crl, y0=y0, R=-R500['radius']) #воздух - линза
bsf17 = Surface(lens_aperture = R500['lens aperture']/2, material_nr = 1.0, x_c=asf17.get_surface_apex(x_c)+d/2, y0=y0, R=R500['radius']) #воздух - линза (Be)

asf18 = Surface(lens_aperture = R500['lens aperture']/2, material_nr = material_n, x_c= bsf17.get_surface_apex(R500['lens aperture']/2) + (bsf17.get_surface_apex(R500['lens aperture']/2) - bsf17.get_surface_apex(x_c)) + dist_crl, y0=y0, R=-R500['radius']) #воздух - линза
bsf18 = Surface(lens_aperture = R500['lens aperture']/2, material_nr = 1.0, x_c=asf18.get_surface_apex(x_c)+d/2, y0=y0, R=R500['radius']) #воздух - линза (Be)

asf19 = Surface(lens_aperture = R500['lens aperture']/2, material_nr = material_n, x_c= bsf18.get_surface_apex(R500['lens aperture']/2) + (bsf18.get_surface_apex(R500['lens aperture']/2) - bsf18.get_surface_apex(x_c)) + dist_crl, y0=y0, R=-R500['radius']) #воздух - линза
bsf19 = Surface(lens_aperture = R500['lens aperture']/2, material_nr = 1.0, x_c=asf19.get_surface_apex(x_c)+d/2, y0=y0, R=R500['radius']) #воздух - линза (Be)

asf20 = Surface(lens_aperture = R500['lens aperture']/2, material_nr = material_n, x_c= bsf19.get_surface_apex(R500['lens aperture']/2) + (bsf19.get_surface_apex(R500['lens aperture']/2) - bsf19.get_surface_apex(x_c)) + dist_crl, y0=y0, R=-R500['radius']) #воздух - линза
bsf20 = Surface(lens_aperture = R500['lens aperture']/2, material_nr = 1.0, x_c=asf20.get_surface_apex(x_c)+d/2, y0=y0, R=R500['radius']) #воздух - линза (Be)

asf21 = Surface(lens_aperture = R500['lens aperture']/2, material_nr = material_n, x_c= bsf20.get_surface_apex(R500['lens aperture']/2) + (bsf20.get_surface_apex(R500['lens aperture']/2) - bsf20.get_surface_apex(x_c)) + dist_crl, y0=y0, R=-R500['radius']) #воздух - линза
bsf21 = Surface(lens_aperture = R500['lens aperture']/2, material_nr = 1.0, x_c=asf21.get_surface_apex(x_c)+d/2, y0=y0, R=R500['radius']) #воздух - линза (Be)

asf22 = Surface(lens_aperture = R500['lens aperture']/2, material_nr = material_n, x_c= bsf21.get_surface_apex(R500['lens aperture']/2) + (bsf21.get_surface_apex(R500['lens aperture']/2) - bsf21.get_surface_apex(x_c)) + dist_crl, y0=y0, R=-R500['radius']) #воздух - линза
bsf22 = Surface(lens_aperture = R500['lens aperture']/2, material_nr = 1.0, x_c=asf22.get_surface_apex(x_c)+d/2, y0=y0, R=R500['radius']) #воздух - линза (Be)

asf23 = Surface(lens_aperture = R500['lens aperture']/2, material_nr = material_n, x_c= bsf22.get_surface_apex(R500['lens aperture']/2) + (bsf22.get_surface_apex(R500['lens aperture']/2) - bsf22.get_surface_apex(x_c)) + dist_crl, y0=y0, R=-R500['radius']) #воздух - линза
bsf23 = Surface(lens_aperture = R500['lens aperture']/2, material_nr = 1.0, x_c=asf23.get_surface_apex(x_c)+d/2, y0=y0, R=R500['radius']) #воздух - линза (Be)

asf24 = Surface(lens_aperture = R500['lens aperture']/2, material_nr = material_n, x_c= bsf23.get_surface_apex(R500['lens aperture']/2) + (bsf23.get_surface_apex(R500['lens aperture']/2) - bsf23.get_surface_apex(x_c)) + dist_crl, y0=y0, R=-R500['radius']) #воздух - линза
bsf24 = Surface(lens_aperture = R500['lens aperture']/2, material_nr = 1.0, x_c=asf24.get_surface_apex(x_c)+d/2, y0=y0, R=R500['radius']) #воздух - линза (Be)

asf25 = Surface(lens_aperture = R500['lens aperture']/2, material_nr = material_n, x_c= bsf24.get_surface_apex(R500['lens aperture']/2) + (bsf24.get_surface_apex(R500['lens aperture']/2) - bsf24.get_surface_apex(x_c)) + dist_crl, y0=y0, R=-R500['radius']) #воздух - линза
bsf25 = Surface(lens_aperture = R500['lens aperture']/2, material_nr = 1.0, x_c=asf25.get_surface_apex(x_c)+d/2, y0=y0, R=R500['radius']) #воздух - линза (Be)

asf26 = Surface(lens_aperture = R500['lens aperture']/2, material_nr = material_n, x_c= bsf25.get_surface_apex(R500['lens aperture']/2) + (bsf25.get_surface_apex(R500['lens aperture']/2) - bsf25.get_surface_apex(x_c)) + dist_crl, y0=y0, R=-R500['radius']) #воздух - линза
bsf26 = Surface(lens_aperture = R500['lens aperture']/2, material_nr = 1.0, x_c=asf26.get_surface_apex(x_c)+d/2, y0=y0, R=R500['radius']) #воздух - линза (Be)

asf27 = Surface(lens_aperture = R500['lens aperture']/2, material_nr = material_n, x_c= bsf26.get_surface_apex(R500['lens aperture']/2) + (bsf26.get_surface_apex(R500['lens aperture']/2) - bsf26.get_surface_apex(x_c)) + dist_crl, y0=y0, R=-R500['radius']) #воздух - линза
bsf27 = Surface(lens_aperture = R500['lens aperture']/2, material_nr = 1.0, x_c=asf27.get_surface_apex(x_c)+d/2, y0=y0, R=R500['radius']) #воздух - линза (Be)

asf28 = Surface(lens_aperture = R500['lens aperture']/2, material_nr = material_n, x_c= bsf27.get_surface_apex(R500['lens aperture']/2) + (bsf27.get_surface_apex(R500['lens aperture']/2) - bsf27.get_surface_apex(x_c)) + dist_crl, y0=y0, R=-R500['radius']) #воздух - линза
bsf28 = Surface(lens_aperture = R500['lens aperture']/2, material_nr = 1.0, x_c=asf28.get_surface_apex(x_c)+d/2, y0=y0, R=R500['radius']) #воздух - линза (Be)

asf29 = Surface(lens_aperture = R500['lens aperture']/2, material_nr = material_n, x_c= bsf28.get_surface_apex(R500['lens aperture']/2) + (bsf28.get_surface_apex(R500['lens aperture']/2) - bsf28.get_surface_apex(x_c)) + dist_crl, y0=y0, R=-R500['radius']) #воздух - линза
bsf29 = Surface(lens_aperture = R500['lens aperture']/2, material_nr = 1.0, x_c=asf29.get_surface_apex(x_c)+d/2, y0=y0, R=R500['radius']) #воздух - линза (Be)

asf30 = Surface(lens_aperture = R500['lens aperture']/2, material_nr = material_n, x_c= bsf29.get_surface_apex(R500['lens aperture']/2) + (bsf29.get_surface_apex(R500['lens aperture']/2) - bsf29.get_surface_apex(x_c)) + dist_crl, y0=y0, R=-R500['radius']) #воздух - линза
bsf30 = Surface(lens_aperture = R500['lens aperture']/2, material_nr = 1.0, x_c=asf30.get_surface_apex(x_c)+d/2, y0=y0, R=R500['radius']) #воздух - линза (Be)

asf31 = Surface(lens_aperture = R500['lens aperture']/2, material_nr = material_n, x_c= bsf30.get_surface_apex(R500['lens aperture']/2) + (bsf30.get_surface_apex(R500['lens aperture']/2) - bsf30.get_surface_apex(x_c)) + dist_crl, y0=y0, R=-R500['radius']) #воздух - линза
bsf31 = Surface(lens_aperture = R500['lens aperture']/2, material_nr = 1.0, x_c=asf31.get_surface_apex(x_c)+d/2, y0=y0, R=R500['radius']) #воздух - линза (Be)

asf32 = Surface(lens_aperture = R500['lens aperture']/2, material_nr = material_n, x_c= bsf31.get_surface_apex(R500['lens aperture']/2) + (bsf31.get_surface_apex(R500['lens aperture']/2) - bsf31.get_surface_apex(x_c)) + dist_crl, y0=y0, R=-R500['radius']) #воздух - линза
bsf32 = Surface(lens_aperture = R500['lens aperture']/2, material_nr = 1.0, x_c=asf32.get_surface_apex(x_c)+d/2, y0=y0, R=R500['radius']) #воздух - линза (Be)

asf33 = Surface(lens_aperture = R500['lens aperture']/2, material_nr = material_n, x_c= bsf32.get_surface_apex(R500['lens aperture']/2) + (bsf32.get_surface_apex(R500['lens aperture']/2) - bsf32.get_surface_apex(x_c)) + dist_crl, y0=y0, R=-R500['radius']) #воздух - линза
bsf33 = Surface(lens_aperture = R500['lens aperture']/2, material_nr = 1.0, x_c=asf33.get_surface_apex(x_c)+d/2, y0=y0, R=R500['radius']) #воздух - линза (Be)

asf34 = Surface(lens_aperture = R500['lens aperture']/2, material_nr = material_n, x_c= bsf33.get_surface_apex(R500['lens aperture']/2) + (bsf33.get_surface_apex(R500['lens aperture']/2) - bsf33.get_surface_apex(x_c)) + dist_crl, y0=y0, R=-R500['radius']) #воздух - линза
bsf34 = Surface(lens_aperture = R500['lens aperture']/2, material_nr = 1.0, x_c=asf34.get_surface_apex(x_c)+d/2, y0=y0, R=R500['radius']) #воздух - линза (Be)

asf35 = Surface(lens_aperture = R500['lens aperture']/2, material_nr = material_n, x_c= bsf34.get_surface_apex(R500['lens aperture']/2) + (bsf34.get_surface_apex(R500['lens aperture']/2) - bsf34.get_surface_apex(x_c)) + dist_crl, y0=y0, R=-R500['radius']) #воздух - линза
bsf35 = Surface(lens_aperture = R500['lens aperture']/2, material_nr = 1.0, x_c=asf35.get_surface_apex(x_c)+d/2, y0=y0, R=R500['radius']) #воздух - линза (Be)

asf36 = Surface(lens_aperture = R500['lens aperture']/2, material_nr = material_n, x_c= bsf35.get_surface_apex(R500['lens aperture']/2) + (bsf35.get_surface_apex(R500['lens aperture']/2) - bsf35.get_surface_apex(x_c)) + dist_crl, y0=y0, R=-R500['radius']) #воздух - линза
bsf36 = Surface(lens_aperture = R500['lens aperture']/2, material_nr = 1.0, x_c=asf36.get_surface_apex(x_c)+d/2, y0=y0, R=R500['radius']) #воздух - линза (Be)

asf37 = Surface(lens_aperture = R500['lens aperture']/2, material_nr = material_n, x_c= bsf36.get_surface_apex(R500['lens aperture']/2) + (bsf36.get_surface_apex(R500['lens aperture']/2) - bsf36.get_surface_apex(x_c)) + dist_crl, y0=y0, R=-R500['radius']) #воздух - линза
bsf37 = Surface(lens_aperture = R500['lens aperture']/2, material_nr = 1.0, x_c=asf37.get_surface_apex(x_c)+d/2, y0=y0, R=R500['radius']) #воздух - линза (Be)

asf38 = Surface(lens_aperture = R500['lens aperture']/2, material_nr = material_n, x_c= bsf37.get_surface_apex(R500['lens aperture']/2) + (bsf37.get_surface_apex(R500['lens aperture']/2) - bsf37.get_surface_apex(x_c)) + dist_crl, y0=y0, R=-R500['radius']) #воздух - линза
bsf38 = Surface(lens_aperture = R500['lens aperture']/2, material_nr = 1.0, x_c=asf38.get_surface_apex(x_c)+d/2, y0=y0, R=R500['radius']) #воздух - линза (Be)

asf39 = Surface(lens_aperture = R500['lens aperture']/2, material_nr = material_n, x_c= bsf38.get_surface_apex(R500['lens aperture']/2) + (bsf38.get_surface_apex(R500['lens aperture']/2) - bsf38.get_surface_apex(x_c)) + dist_crl, y0=y0, R=-R500['radius']) #воздух - линза
bsf39 = Surface(lens_aperture = R500['lens aperture']/2, material_nr = 1.0, x_c=asf39.get_surface_apex(x_c)+d/2, y0=y0, R=R500['radius']) #воздух - линза (Be)

asf40 = Surface(lens_aperture = R500['lens aperture']/2, material_nr = material_n, x_c= bsf39.get_surface_apex(R500['lens aperture']/2) + (bsf39.get_surface_apex(R500['lens aperture']/2) - bsf39.get_surface_apex(x_c)) + dist_crl, y0=y0, R=-R500['radius']) #воздух - линза
bsf40 = Surface(lens_aperture = R500['lens aperture']/2, material_nr = 1.0, x_c=asf40.get_surface_apex(x_c)+d/2, y0=y0, R=R500['radius']) #воздух - линза (Be)

asf41 = Surface(lens_aperture = R500['lens aperture']/2, material_nr = material_n, x_c= bsf40.get_surface_apex(R500['lens aperture']/2) + (bsf40.get_surface_apex(R500['lens aperture']/2) - bsf40.get_surface_apex(x_c)) + dist_crl, y0=y0, R=-R500['radius']) #воздух - линза
bsf41 = Surface(lens_aperture = R500['lens aperture']/2, material_nr = 1.0, x_c=asf41.get_surface_apex(x_c)+d/2, y0=y0, R=R500['radius']) #воздух - линза (Be)

asf42 = Surface(lens_aperture = R500['lens aperture']/2, material_nr = material_n, x_c= bsf41.get_surface_apex(R500['lens aperture']/2) + (bsf41.get_surface_apex(R500['lens aperture']/2) - bsf41.get_surface_apex(x_c)) + dist_crl, y0=y0, R=-R500['radius']) #воздух - линза
bsf42 = Surface(lens_aperture = R500['lens aperture']/2, material_nr = 1.0, x_c=asf42.get_surface_apex(x_c)+d/2, y0=y0, R=R500['radius']) #воздух - линза (Be)

asf43 = Surface(lens_aperture = R500['lens aperture']/2, material_nr = material_n, x_c= bsf42.get_surface_apex(R500['lens aperture']/2) + (bsf42.get_surface_apex(R500['lens aperture']/2) - bsf42.get_surface_apex(x_c)) + dist_crl, y0=y0, R=-R500['radius']) #воздух - линза
bsf43 = Surface(lens_aperture = R500['lens aperture']/2, material_nr = 1.0, x_c=asf43.get_surface_apex(x_c)+d/2, y0=y0, R=R500['radius']) #воздух - линза (Be)

asf44 = Surface(lens_aperture = R500['lens aperture']/2, material_nr = material_n, x_c= bsf43.get_surface_apex(R500['lens aperture']/2) + (bsf43.get_surface_apex(R500['lens aperture']/2) - bsf43.get_surface_apex(x_c)) + dist_crl, y0=y0, R=-R500['radius']) #воздух - линза
bsf44 = Surface(lens_aperture = R500['lens aperture']/2, material_nr = 1.0, x_c=asf44.get_surface_apex(x_c)+d/2, y0=y0, R=R500['radius']) #воздух - линза (Be)

asf45 = Surface(lens_aperture = R500['lens aperture']/2, material_nr = material_n, x_c= bsf44.get_surface_apex(R500['lens aperture']/2) + (bsf44.get_surface_apex(R500['lens aperture']/2) - bsf44.get_surface_apex(x_c)) + dist_crl, y0=y0, R=-R500['radius']) #воздух - линза
bsf45 = Surface(lens_aperture = R500['lens aperture']/2, material_nr = 1.0, x_c=asf45.get_surface_apex(x_c)+d/2, y0=y0, R=R500['radius']) #воздух - линза (Be)

asf46 = Surface(lens_aperture = R500['lens aperture']/2, material_nr = material_n, x_c= bsf45.get_surface_apex(R500['lens aperture']/2) + (bsf45.get_surface_apex(R500['lens aperture']/2) - bsf45.get_surface_apex(x_c)) + dist_crl, y0=y0, R=-R500['radius']) #воздух - линза
bsf46 = Surface(lens_aperture = R500['lens aperture']/2, material_nr = 1.0, x_c=asf46.get_surface_apex(x_c)+d/2, y0=y0, R=R500['radius']) #воздух - линза (Be)

asf47 = Surface(lens_aperture = R500['lens aperture']/2, material_nr = material_n, x_c= bsf46.get_surface_apex(R500['lens aperture']/2) + (bsf46.get_surface_apex(R500['lens aperture']/2) - bsf46.get_surface_apex(x_c)) + dist_crl, y0=y0, R=-R500['radius']) #воздух - линза
bsf47 = Surface(lens_aperture = R500['lens aperture']/2, material_nr = 1.0, x_c=asf47.get_surface_apex(x_c)+d/2, y0=y0, R=R500['radius']) #воздух - линза (Be)

asf48 = Surface(lens_aperture = R500['lens aperture']/2, material_nr = material_n, x_c= bsf47.get_surface_apex(R500['lens aperture']/2) + (bsf47.get_surface_apex(R500['lens aperture']/2) - bsf47.get_surface_apex(x_c)) + dist_crl, y0=y0, R=-R500['radius']) #воздух - линза
bsf48 = Surface(lens_aperture = R500['lens aperture']/2, material_nr = 1.0, x_c=asf48.get_surface_apex(x_c)+d/2, y0=y0, R=R500['radius']) #воздух - линза (Be)

asf49 = Surface(lens_aperture = R500['lens aperture']/2, material_nr = material_n, x_c= bsf48.get_surface_apex(R500['lens aperture']/2) + (bsf48.get_surface_apex(R500['lens aperture']/2) - bsf48.get_surface_apex(x_c)) + dist_crl, y0=y0, R=-R500['radius']) #воздух - линза
bsf49 = Surface(lens_aperture = R500['lens aperture']/2, material_nr = 1.0, x_c=asf49.get_surface_apex(x_c)+d/2, y0=y0, R=R500['radius']) #воздух - линза (Be)

asf50 = Surface(lens_aperture = R500['lens aperture']/2, material_nr = material_n, x_c= bsf49.get_surface_apex(R500['lens aperture']/2) + (bsf49.get_surface_apex(R500['lens aperture']/2) - bsf49.get_surface_apex(x_c)) + dist_crl, y0=y0, R=-R500['radius']) #воздух - линза
bsf50 = Surface(lens_aperture = R500['lens aperture']/2, material_nr = 1.0, x_c=asf50.get_surface_apex(x_c)+d/2, y0=y0, R=R500['radius']) #воздух - линза (Be)

asf51 = Surface(lens_aperture = R500['lens aperture']/2, material_nr = material_n, x_c= bsf50.get_surface_apex(R500['lens aperture']/2) + (bsf50.get_surface_apex(R500['lens aperture']/2) - bsf50.get_surface_apex(x_c)) + dist_crl, y0=y0, R=-R500['radius']) #воздух - линза
bsf51 = Surface(lens_aperture = R500['lens aperture']/2, material_nr = 1.0, x_c=asf51.get_surface_apex(x_c)+d/2, y0=y0, R=R500['radius']) #воздух - линза (Be)

asf52 = Surface(lens_aperture = R500['lens aperture']/2, material_nr = material_n, x_c= bsf51.get_surface_apex(R500['lens aperture']/2) + (bsf51.get_surface_apex(R500['lens aperture']/2) - bsf51.get_surface_apex(x_c)) + dist_crl, y0=y0, R=-R500['radius']) #воздух - линза
bsf52 = Surface(lens_aperture = R500['lens aperture']/2, material_nr = 1.0, x_c=asf52.get_surface_apex(x_c)+d/2, y0=y0, R=R500['radius']) #воздух - линза (Be)

asf53 = Surface(lens_aperture = R500['lens aperture']/2, material_nr = material_n, x_c= bsf52.get_surface_apex(R500['lens aperture']/2) + (bsf52.get_surface_apex(R500['lens aperture']/2) - bsf52.get_surface_apex(x_c)) + dist_crl, y0=y0, R=-R500['radius']) #воздух - линза
bsf53 = Surface(lens_aperture = R500['lens aperture']/2, material_nr = 1.0, x_c=asf53.get_surface_apex(x_c)+d/2, y0=y0, R=R500['radius']) #воздух - линза (Be)

asf54 = Surface(lens_aperture = R500['lens aperture']/2, material_nr = material_n, x_c= bsf53.get_surface_apex(R500['lens aperture']/2) + (bsf53.get_surface_apex(R500['lens aperture']/2) - bsf53.get_surface_apex(x_c)) + dist_crl, y0=y0, R=-R500['radius']) #воздух - линза
bsf54 = Surface(lens_aperture = R500['lens aperture']/2, material_nr = 1.0, x_c=asf54.get_surface_apex(x_c)+d/2, y0=y0, R=R500['radius']) #воздух - линза (Be)

asf55 = Surface(lens_aperture = R500['lens aperture']/2, material_nr = material_n, x_c= bsf54.get_surface_apex(R500['lens aperture']/2) + (bsf54.get_surface_apex(R500['lens aperture']/2) - bsf54.get_surface_apex(x_c)) + dist_crl, y0=y0, R=-R500['radius']) #воздух - линза
bsf55 = Surface(lens_aperture = R500['lens aperture']/2, material_nr = 1.0, x_c=asf55.get_surface_apex(x_c)+d/2, y0=y0, R=R500['radius']) #воздух - линза (Be)

asf56 = Surface(lens_aperture = R500['lens aperture']/2, material_nr = material_n, x_c= bsf55.get_surface_apex(R500['lens aperture']/2) + (bsf55.get_surface_apex(R500['lens aperture']/2) - bsf55.get_surface_apex(x_c)) + dist_crl, y0=y0, R=-R500['radius']) #воздух - линза
bsf56 = Surface(lens_aperture = R500['lens aperture']/2, material_nr = 1.0, x_c=asf56.get_surface_apex(x_c)+d/2, y0=y0, R=R500['radius']) #воздух - линза (Be)

asf57 = Surface(lens_aperture = R500['lens aperture']/2, material_nr = material_n, x_c= bsf56.get_surface_apex(R500['lens aperture']/2) + (bsf56.get_surface_apex(R500['lens aperture']/2) - bsf56.get_surface_apex(x_c)) + dist_crl, y0=y0, R=-R500['radius']) #воздух - линза
bsf57 = Surface(lens_aperture = R500['lens aperture']/2, material_nr = 1.0, x_c=asf57.get_surface_apex(x_c)+d/2, y0=y0, R=R500['radius']) #воздух - линза (Be)

asf58 = Surface(lens_aperture = R500['lens aperture']/2, material_nr = material_n, x_c= bsf57.get_surface_apex(R500['lens aperture']/2) + (bsf57.get_surface_apex(R500['lens aperture']/2) - bsf57.get_surface_apex(x_c)) + dist_crl, y0=y0, R=-R500['radius']) #воздух - линза
bsf58 = Surface(lens_aperture = R500['lens aperture']/2, material_nr = 1.0, x_c=asf58.get_surface_apex(x_c)+d/2, y0=y0, R=R500['radius']) #воздух - линза (Be)

asf59 = Surface(lens_aperture = R500['lens aperture']/2, material_nr = material_n, x_c= bsf58.get_surface_apex(R500['lens aperture']/2) + (bsf58.get_surface_apex(R500['lens aperture']/2) - bsf58.get_surface_apex(x_c)) + dist_crl, y0=y0, R=-R500['radius']) #воздух - линза
bsf59 = Surface(lens_aperture = R500['lens aperture']/2, material_nr = 1.0, x_c=asf59.get_surface_apex(x_c)+d/2, y0=y0, R=R500['radius']) #воздух - линза (Be)

asf60 = Surface(lens_aperture = R500['lens aperture']/2, material_nr = material_n, x_c= bsf59.get_surface_apex(R500['lens aperture']/2) + (bsf59.get_surface_apex(R500['lens aperture']/2) - bsf59.get_surface_apex(x_c)) + dist_crl, y0=y0, R=-R500['radius']) #воздух - линза
bsf60 = Surface(lens_aperture = R500['lens aperture']/2, material_nr = 1.0, x_c=asf60.get_surface_apex(x_c)+d/2, y0=y0, R=R500['radius']) #воздух - линза (Be)



print (bsf1.get_surface_apex(R500['lens aperture']/2) - asf1.get_surface_apex(R500['lens aperture']/2))
print (bsf2.get_surface_apex(x_c) - asf2.get_surface_apex(x_c))


'''
#Plot single lens
fig, ax = plt.subplots()
fig.set_figheight(10)
fig.set_figwidth(15)

asf.render(ax)
bsf.render(ax)
'''

#TF2
asf101 = Surface(lens_aperture = R50['lens aperture']/2, material_nr = material_n, x_c = 5*1E-3 - d/2, y0=y0, R=-R50['radius']) #воздух - линза
bsf101 = Surface(lens_aperture = R50['lens aperture']/2, material_nr = 1.0, x_c=asf101.get_surface_apex(x_c)+d, y0=y0, R=R50['radius']) #воздух - линза (Be)

asf102 = Surface(lens_aperture = R50['lens aperture']/2, material_nr = material_n, x_c = bsf101.get_surface_apex(R50['lens aperture']/2) + (bsf101.get_surface_apex(R50['lens aperture']/2) - bsf101.get_surface_apex(x_c)) + dist_crl, y0=y0, R=-R50['radius']) #воздух - линза
bsf102 = Surface(lens_aperture = R50['lens aperture']/2, material_nr = 1.0, x_c=asf102.get_surface_apex(x_c)+d, y0=y0, R=R50['radius']) #воздух - линза (Be)

asf103 = Surface(lens_aperture = R50['lens aperture']/2, material_nr = material_n, x_c = bsf102.get_surface_apex(R50['lens aperture']/2) + (bsf102.get_surface_apex(R50['lens aperture']/2) - bsf102.get_surface_apex(x_c)) + dist_crl, y0=y0, R=-R50['radius']) #воздух - линза
bsf103 = Surface(lens_aperture = R50['lens aperture']/2, material_nr = 1.0, x_c=asf103.get_surface_apex(x_c)+d, y0=y0, R=R50['radius']) #воздух - линза (Be)

asf104 = Surface(lens_aperture = R50['lens aperture']/2, material_nr = material_n, x_c = bsf103.get_surface_apex(R50['lens aperture']/2) + (bsf103.get_surface_apex(R50['lens aperture']/2) - bsf103.get_surface_apex(x_c)) + dist_crl, y0=y0, R=-R50['radius']) #воздух - линза
bsf104 = Surface(lens_aperture = R50['lens aperture']/2, material_nr = 1.0, x_c=asf104.get_surface_apex(x_c)+d, y0=y0, R=R50['radius']) #воздух - линза (Be)

asf105 = Surface(lens_aperture = R50['lens aperture']/2, material_nr = material_n, x_c = bsf104.get_surface_apex(R50['lens aperture']/2) + (bsf104.get_surface_apex(R50['lens aperture']/2) - bsf104.get_surface_apex(x_c)) + dist_crl, y0=y0, R=-R50['radius']) #воздух - линза
bsf105 = Surface(lens_aperture = R50['lens aperture']/2, material_nr = 1.0, x_c=asf105.get_surface_apex(x_c)+d, y0=y0, R=R50['radius']) #воздух - линза (Be)

asf106 = Surface(lens_aperture = R50['lens aperture']/2, material_nr = material_n, x_c = bsf105.get_surface_apex(R50['lens aperture']/2) + (bsf105.get_surface_apex(R50['lens aperture']/2) - bsf105.get_surface_apex(x_c)) + dist_crl, y0=y0, R=-R50['radius']) #воздух - линза
bsf106 = Surface(lens_aperture = R50['lens aperture']/2, material_nr = 1.0, x_c=asf106.get_surface_apex(x_c)+d, y0=y0, R=R50['radius']) #воздух - линза (Be)

asf107 = Surface(lens_aperture = R50['lens aperture']/2, material_nr = material_n, x_c = bsf106.get_surface_apex(R50['lens aperture']/2) + (bsf106.get_surface_apex(R50['lens aperture']/2) - bsf106.get_surface_apex(x_c)) + dist_crl, y0=y0, R=-R50['radius']) #воздух - линза
bsf107 = Surface(lens_aperture = R50['lens aperture']/2, material_nr = 1.0, x_c=asf107.get_surface_apex(x_c)+d, y0=y0, R=R50['radius']) #воздух - линза (Be)

asf108 = Surface(lens_aperture = R50['lens aperture']/2, material_nr = material_n, x_c = bsf107.get_surface_apex(R50['lens aperture']/2) + (bsf107.get_surface_apex(R50['lens aperture']/2) - bsf107.get_surface_apex(x_c)) + dist_crl, y0=y0, R=-R50['radius']) #воздух - линза
bsf108 = Surface(lens_aperture = R50['lens aperture']/2, material_nr = 1.0, x_c=asf108.get_surface_apex(x_c)+d, y0=y0, R=R50['radius']) #воздух - линза (Be)

asf109 = Surface(lens_aperture = R50['lens aperture']/2, material_nr = material_n, x_c = bsf108.get_surface_apex(R50['lens aperture']/2) + (bsf108.get_surface_apex(R50['lens aperture']/2) - bsf108.get_surface_apex(x_c)) + dist_crl, y0=y0, R=-R50['radius']) #воздух - линза
bsf109 = Surface(lens_aperture = R50['lens aperture']/2, material_nr = 1.0, x_c=asf109.get_surface_apex(x_c)+d, y0=y0, R=R50['radius']) #воздух - линза (Be)

asf110 = Surface(lens_aperture = R50['lens aperture']/2, material_nr = material_n, x_c = bsf109.get_surface_apex(R50['lens aperture']/2) + (bsf109.get_surface_apex(R50['lens aperture']/2) - bsf109.get_surface_apex(x_c)) + dist_crl, y0=y0, R=-R50['radius']) #воздух - линза
bsf110 = Surface(lens_aperture = R50['lens aperture']/2, material_nr = 1.0, x_c=asf110.get_surface_apex(x_c)+d, y0=y0, R=R50['radius']) #воздух - линза (Be)

asf111 = Surface(lens_aperture = R50['lens aperture']/2, material_nr = material_n, x_c = bsf110.get_surface_apex(R50['lens aperture']/2) + (bsf110.get_surface_apex(R50['lens aperture']/2) - bsf110.get_surface_apex(x_c)) + dist_crl, y0=y0, R=-R50['radius']) #воздух - линза
bsf111 = Surface(lens_aperture = R50['lens aperture']/2, material_nr = 1.0, x_c=asf111.get_surface_apex(x_c)+d, y0=y0, R=R50['radius']) #воздух - линза (Be)

asf112 = Surface(lens_aperture = R50['lens aperture']/2, material_nr = material_n, x_c = bsf111.get_surface_apex(R50['lens aperture']/2) + (bsf111.get_surface_apex(R50['lens aperture']/2) - bsf111.get_surface_apex(x_c)) + dist_crl, y0=y0, R=-R50['radius']) #воздух - линза
bsf112 = Surface(lens_aperture = R50['lens aperture']/2, material_nr = 1.0, x_c=asf112.get_surface_apex(x_c)+d, y0=y0, R=R50['radius']) #воздух - линза (Be)

asf113 = Surface(lens_aperture = R50['lens aperture']/2, material_nr = material_n, x_c = bsf112.get_surface_apex(R50['lens aperture']/2) + (bsf112.get_surface_apex(R50['lens aperture']/2) - bsf112.get_surface_apex(x_c)) + dist_crl, y0=y0, R=-R50['radius']) #воздух - линза
bsf113 = Surface(lens_aperture = R50['lens aperture']/2, material_nr = 1.0, x_c=asf113.get_surface_apex(x_c)+d, y0=y0, R=R50['radius']) #воздух - линза (Be)

asf114 = Surface(lens_aperture = R50['lens aperture']/2, material_nr = material_n, x_c = bsf113.get_surface_apex(R50['lens aperture']/2) + (bsf113.get_surface_apex(R50['lens aperture']/2) - bsf113.get_surface_apex(x_c)) + dist_crl, y0=y0, R=-R50['radius']) #воздух - линза
bsf114 = Surface(lens_aperture = R50['lens aperture']/2, material_nr = 1.0, x_c=asf114.get_surface_apex(x_c)+d, y0=y0, R=R50['radius']) #воздух - линза (Be)

asf115 = Surface(lens_aperture = R50['lens aperture']/2, material_nr = material_n, x_c = bsf114.get_surface_apex(R50['lens aperture']/2) + (bsf114.get_surface_apex(R50['lens aperture']/2) - bsf114.get_surface_apex(x_c)) + dist_crl, y0=y0, R=-R50['radius']) #воздух - линза
bsf115 = Surface(lens_aperture = R50['lens aperture']/2, material_nr = 1.0, x_c=asf115.get_surface_apex(x_c)+d, y0=y0, R=R50['radius']) #воздух - линза (Be)

asf116 = Surface(lens_aperture = R50['lens aperture']/2, material_nr = material_n, x_c = bsf115.get_surface_apex(R50['lens aperture']/2) + (bsf115.get_surface_apex(R50['lens aperture']/2) - bsf115.get_surface_apex(x_c)) + dist_crl, y0=y0, R=-R50['radius']) #воздух - линза
bsf116 = Surface(lens_aperture = R50['lens aperture']/2, material_nr = 1.0, x_c=asf116.get_surface_apex(x_c)+d, y0=y0, R=R50['radius']) #воздух - линза (Be)

asf117 = Surface(lens_aperture = R50['lens aperture']/2, material_nr = material_n, x_c = bsf116.get_surface_apex(R50['lens aperture']/2) + (bsf116.get_surface_apex(R50['lens aperture']/2) - bsf116.get_surface_apex(x_c)) + dist_crl, y0=y0, R=-R50['radius']) #воздух - линза
bsf117 = Surface(lens_aperture = R50['lens aperture']/2, material_nr = 1.0, x_c=asf117.get_surface_apex(x_c)+d, y0=y0, R=R50['radius']) #воздух - линза (Be)

asf118 = Surface(lens_aperture = R50['lens aperture']/2, material_nr = material_n, x_c = bsf117.get_surface_apex(R50['lens aperture']/2) + (bsf117.get_surface_apex(R50['lens aperture']/2) - bsf117.get_surface_apex(x_c)) + dist_crl, y0=y0, R=-R50['radius']) #воздух - линза
bsf118 = Surface(lens_aperture = R50['lens aperture']/2, material_nr = 1.0, x_c=asf118.get_surface_apex(x_c)+d, y0=y0, R=R50['radius']) #воздух - линза (Be)

asf119 = Surface(lens_aperture = R50['lens aperture']/2, material_nr = material_n, x_c = bsf118.get_surface_apex(R50['lens aperture']/2) + (bsf118.get_surface_apex(R50['lens aperture']/2) - bsf118.get_surface_apex(x_c)) + dist_crl, y0=y0, R=-R50['radius']) #воздух - линза
bsf119 = Surface(lens_aperture = R50['lens aperture']/2, material_nr = 1.0, x_c=asf119.get_surface_apex(x_c)+d, y0=y0, R=R50['radius']) #воздух - линза (Be)

asf120 = Surface(lens_aperture = R50['lens aperture']/2, material_nr = material_n, x_c = bsf119.get_surface_apex(R50['lens aperture']/2) + (bsf119.get_surface_apex(R50['lens aperture']/2) - bsf119.get_surface_apex(x_c)) + dist_crl, y0=y0, R=-R50['radius']) #воздух - линза
bsf120 = Surface(lens_aperture = R50['lens aperture']/2, material_nr = 1.0, x_c=asf120.get_surface_apex(x_c)+d, y0=y0, R=R50['radius']) #воздух - линза (Be)

asf121 = Surface(lens_aperture = R50['lens aperture']/2, material_nr = material_n, x_c = bsf120.get_surface_apex(R50['lens aperture']/2) + (bsf120.get_surface_apex(R50['lens aperture']/2) - bsf120.get_surface_apex(x_c)) + dist_crl, y0=y0, R=-R50['radius']) #воздух - линза
bsf121 = Surface(lens_aperture = R50['lens aperture']/2, material_nr = 1.0, x_c=asf121.get_surface_apex(x_c)+d, y0=y0, R=R50['radius']) #воздух - линза (Be)

asf122 = Surface(lens_aperture = R50['lens aperture']/2, material_nr = material_n, x_c = bsf121.get_surface_apex(R50['lens aperture']/2) + (bsf121.get_surface_apex(R50['lens aperture']/2) - bsf121.get_surface_apex(x_c)) + dist_crl, y0=y0, R=-R50['radius']) #воздух - линза
bsf122 = Surface(lens_aperture = R50['lens aperture']/2, material_nr = 1.0, x_c=asf122.get_surface_apex(x_c)+d, y0=y0, R=R50['radius']) #воздух - линза (Be)

asf123 = Surface(lens_aperture = R50['lens aperture']/2, material_nr = material_n, x_c = bsf122.get_surface_apex(R50['lens aperture']/2) + (bsf122.get_surface_apex(R50['lens aperture']/2) - bsf122.get_surface_apex(x_c)) + dist_crl, y0=y0, R=-R50['radius']) #воздух - линза
bsf123 = Surface(lens_aperture = R50['lens aperture']/2, material_nr = 1.0, x_c=asf123.get_surface_apex(x_c)+d, y0=y0, R=R50['radius']) #воздух - линза (Be)

asf124 = Surface(lens_aperture = R50['lens aperture']/2, material_nr = material_n, x_c = bsf123.get_surface_apex(R50['lens aperture']/2) + (bsf123.get_surface_apex(R50['lens aperture']/2) - bsf123.get_surface_apex(x_c)) + dist_crl, y0=y0, R=-R50['radius']) #воздух - линза
bsf124 = Surface(lens_aperture = R50['lens aperture']/2, material_nr = 1.0, x_c=asf124.get_surface_apex(x_c)+d, y0=y0, R=R50['radius']) #воздух - линза (Be)

asf125 = Surface(lens_aperture = R50['lens aperture']/2, material_nr = material_n, x_c = bsf124.get_surface_apex(R50['lens aperture']/2) + (bsf124.get_surface_apex(R50['lens aperture']/2) - bsf124.get_surface_apex(x_c)) + dist_crl, y0=y0, R=-R50['radius']) #воздух - линза
bsf125 = Surface(lens_aperture = R50['lens aperture']/2, material_nr = 1.0, x_c=asf125.get_surface_apex(x_c)+d, y0=y0, R=R50['radius']) #воздух - линза (Be)

asf126 = Surface(lens_aperture = R50['lens aperture']/2, material_nr = material_n, x_c = bsf125.get_surface_apex(R50['lens aperture']/2) + (bsf125.get_surface_apex(R50['lens aperture']/2) - bsf125.get_surface_apex(x_c)) + dist_crl, y0=y0, R=-R50['radius']) #воздух - линза
bsf126 = Surface(lens_aperture = R50['lens aperture']/2, material_nr = 1.0, x_c=asf126.get_surface_apex(x_c)+d, y0=y0, R=R50['radius']) #воздух - линза (Be)

asf127 = Surface(lens_aperture = R50['lens aperture']/2, material_nr = material_n, x_c = bsf126.get_surface_apex(R50['lens aperture']/2) + (bsf126.get_surface_apex(R50['lens aperture']/2) - bsf126.get_surface_apex(x_c)) + dist_crl, y0=y0, R=-R50['radius']) #воздух - линза
bsf127 = Surface(lens_aperture = R50['lens aperture']/2, material_nr = 1.0, x_c=asf127.get_surface_apex(x_c)+d, y0=y0, R=R50['radius']) #воздух - линза (Be)

asf128 = Surface(lens_aperture = R50['lens aperture']/2, material_nr = material_n, x_c = bsf127.get_surface_apex(R50['lens aperture']/2) + (bsf127.get_surface_apex(R50['lens aperture']/2) - bsf127.get_surface_apex(x_c)) + dist_crl, y0=y0, R=-R50['radius']) #воздух - линза
bsf128 = Surface(lens_aperture = R50['lens aperture']/2, material_nr = 1.0, x_c=asf128.get_surface_apex(x_c)+d, y0=y0, R=R50['radius']) #воздух - линза (Be)

asf129 = Surface(lens_aperture = R50['lens aperture']/2, material_nr = material_n, x_c = bsf128.get_surface_apex(R50['lens aperture']/2) + (bsf128.get_surface_apex(R50['lens aperture']/2) - bsf128.get_surface_apex(x_c)) + dist_crl, y0=y0, R=-R50['radius']) #воздух - линза
bsf129 = Surface(lens_aperture = R50['lens aperture']/2, material_nr = 1.0, x_c=asf129.get_surface_apex(x_c)+d, y0=y0, R=R50['radius']) #воздух - линза (Be)

asf130 = Surface(lens_aperture = R50['lens aperture']/2, material_nr = material_n, x_c = bsf129.get_surface_apex(R50['lens aperture']/2) + (bsf129.get_surface_apex(R50['lens aperture']/2) - bsf129.get_surface_apex(x_c)) + dist_crl, y0=y0, R=-R50['radius']) #воздух - линза
bsf130 = Surface(lens_aperture = R50['lens aperture']/2, material_nr = 1.0, x_c=asf130.get_surface_apex(x_c)+d, y0=y0, R=R50['radius']) #воздух - линза (Be)

asf131 = Surface(lens_aperture = R50['lens aperture']/2, material_nr = material_n, x_c = bsf130.get_surface_apex(R50['lens aperture']/2) + (bsf130.get_surface_apex(R50['lens aperture']/2) - bsf130.get_surface_apex(x_c)) + dist_crl, y0=y0, R=-R50['radius']) #воздух - линза
bsf131 = Surface(lens_aperture = R50['lens aperture']/2, material_nr = 1.0, x_c=asf131.get_surface_apex(x_c)+d, y0=y0, R=R50['radius']) #воздух - линза (Be)

asf132 = Surface(lens_aperture = R50['lens aperture']/2, material_nr = material_n, x_c = bsf131.get_surface_apex(R50['lens aperture']/2) + (bsf131.get_surface_apex(R50['lens aperture']/2) - bsf131.get_surface_apex(x_c)) + dist_crl, y0=y0, R=-R50['radius']) #воздух - линза
bsf132 = Surface(lens_aperture = R50['lens aperture']/2, material_nr = 1.0, x_c=asf132.get_surface_apex(x_c)+d, y0=y0, R=R50['radius']) #воздух - линза (Be)

asf133 = Surface(lens_aperture = R50['lens aperture']/2, material_nr = material_n, x_c = bsf132.get_surface_apex(R50['lens aperture']/2) + (bsf132.get_surface_apex(R50['lens aperture']/2) - bsf132.get_surface_apex(x_c)) + dist_crl, y0=y0, R=-R50['radius']) #воздух - линза
bsf133 = Surface(lens_aperture = R50['lens aperture']/2, material_nr = 1.0, x_c=asf133.get_surface_apex(x_c)+d, y0=y0, R=R50['radius']) #воздух - линза (Be)

asf134 = Surface(lens_aperture = R50['lens aperture']/2, material_nr = material_n, x_c = bsf133.get_surface_apex(R50['lens aperture']/2) + (bsf133.get_surface_apex(R50['lens aperture']/2) - bsf133.get_surface_apex(x_c)) + dist_crl, y0=y0, R=-R50['radius']) #воздух - линза
bsf134 = Surface(lens_aperture = R50['lens aperture']/2, material_nr = 1.0, x_c=asf134.get_surface_apex(x_c)+d, y0=y0, R=R50['radius']) #воздух - линза (Be)

asf135 = Surface(lens_aperture = R50['lens aperture']/2, material_nr = material_n, x_c = bsf134.get_surface_apex(R50['lens aperture']/2) + (bsf134.get_surface_apex(R50['lens aperture']/2) - bsf134.get_surface_apex(x_c)) + dist_crl, y0=y0, R=-R50['radius']) #воздух - линза
bsf135 = Surface(lens_aperture = R50['lens aperture']/2, material_nr = 1.0, x_c=asf135.get_surface_apex(x_c)+d, y0=y0, R=R50['radius']) #воздух - линза (Be)

asf136 = Surface(lens_aperture = R50['lens aperture']/2, material_nr = material_n, x_c = bsf135.get_surface_apex(R50['lens aperture']/2) + (bsf135.get_surface_apex(R50['lens aperture']/2) - bsf135.get_surface_apex(x_c)) + dist_crl, y0=y0, R=-R50['radius']) #воздух - линза
bsf136 = Surface(lens_aperture = R50['lens aperture']/2, material_nr = 1.0, x_c=asf136.get_surface_apex(x_c)+d, y0=y0, R=R50['radius']) #воздух - линза (Be)

asf137 = Surface(lens_aperture = R50['lens aperture']/2, material_nr = material_n, x_c = bsf136.get_surface_apex(R50['lens aperture']/2) + (bsf136.get_surface_apex(R50['lens aperture']/2) - bsf136.get_surface_apex(x_c)) + dist_crl, y0=y0, R=-R50['radius']) #воздух - линза
bsf137 = Surface(lens_aperture = R50['lens aperture']/2, material_nr = 1.0, x_c=asf137.get_surface_apex(x_c)+d, y0=y0, R=R50['radius']) #воздух - линза (Be)

asf138 = Surface(lens_aperture = R50['lens aperture']/2, material_nr = material_n, x_c = bsf137.get_surface_apex(R50['lens aperture']/2) + (bsf137.get_surface_apex(R50['lens aperture']/2) - bsf137.get_surface_apex(x_c)) + dist_crl, y0=y0, R=-R50['radius']) #воздух - линза
bsf138 = Surface(lens_aperture = R50['lens aperture']/2, material_nr = 1.0, x_c=asf138.get_surface_apex(x_c)+d, y0=y0, R=R50['radius']) #воздух - линза (Be)

asf139 = Surface(lens_aperture = R50['lens aperture']/2, material_nr = material_n, x_c = bsf138.get_surface_apex(R50['lens aperture']/2) + (bsf138.get_surface_apex(R50['lens aperture']/2) - bsf138.get_surface_apex(x_c)) + dist_crl, y0=y0, R=-R50['radius']) #воздух - линза
bsf139 = Surface(lens_aperture = R50['lens aperture']/2, material_nr = 1.0, x_c=asf139.get_surface_apex(x_c)+d, y0=y0, R=R50['radius']) #воздух - линза (Be)

asf140 = Surface(lens_aperture = R50['lens aperture']/2, material_nr = material_n, x_c = bsf139.get_surface_apex(R50['lens aperture']/2) + (bsf139.get_surface_apex(R50['lens aperture']/2) - bsf139.get_surface_apex(x_c)) + dist_crl, y0=y0, R=-R50['radius']) #воздух - линза
bsf140 = Surface(lens_aperture = R50['lens aperture']/2, material_nr = 1.0, x_c=asf140.get_surface_apex(x_c)+d, y0=y0, R=R50['radius']) #воздух - линза (Be)

asf141 = Surface(lens_aperture = R50['lens aperture']/2, material_nr = material_n, x_c = bsf140.get_surface_apex(R50['lens aperture']/2) + (bsf140.get_surface_apex(R50['lens aperture']/2) - bsf140.get_surface_apex(x_c)) + dist_crl, y0=y0, R=-R50['radius']) #воздух - линза
bsf141 = Surface(lens_aperture = R50['lens aperture']/2, material_nr = 1.0, x_c=asf141.get_surface_apex(x_c)+d, y0=y0, R=R50['radius']) #воздух - линза (Be)

asf142 = Surface(lens_aperture = R50['lens aperture']/2, material_nr = material_n, x_c = bsf141.get_surface_apex(R50['lens aperture']/2) + (bsf141.get_surface_apex(R50['lens aperture']/2) - bsf141.get_surface_apex(x_c)) + dist_crl, y0=y0, R=-R50['radius']) #воздух - линза
bsf142 = Surface(lens_aperture = R50['lens aperture']/2, material_nr = 1.0, x_c=asf142.get_surface_apex(x_c)+d, y0=y0, R=R50['radius']) #воздух - линза (Be)

asf143 = Surface(lens_aperture = R50['lens aperture']/2, material_nr = material_n, x_c = bsf142.get_surface_apex(R50['lens aperture']/2) + (bsf142.get_surface_apex(R50['lens aperture']/2) - bsf142.get_surface_apex(x_c)) + dist_crl, y0=y0, R=-R50['radius']) #воздух - линза
bsf143 = Surface(lens_aperture = R50['lens aperture']/2, material_nr = 1.0, x_c=asf143.get_surface_apex(x_c)+d, y0=y0, R=R50['radius']) #воздух - линза (Be)

asf144 = Surface(lens_aperture = R50['lens aperture']/2, material_nr = material_n, x_c = bsf143.get_surface_apex(R50['lens aperture']/2) + (bsf143.get_surface_apex(R50['lens aperture']/2) - bsf143.get_surface_apex(x_c)) + dist_crl, y0=y0, R=-R50['radius']) #воздух - линза
bsf144 = Surface(lens_aperture = R50['lens aperture']/2, material_nr = 1.0, x_c=asf144.get_surface_apex(x_c)+d, y0=y0, R=R50['radius']) #воздух - линза (Be)

asf145 = Surface(lens_aperture = R50['lens aperture']/2, material_nr = material_n, x_c = bsf144.get_surface_apex(R50['lens aperture']/2) + (bsf144.get_surface_apex(R50['lens aperture']/2) - bsf144.get_surface_apex(x_c)) + dist_crl, y0=y0, R=-R50['radius']) #воздух - линза
bsf145 = Surface(lens_aperture = R50['lens aperture']/2, material_nr = 1.0, x_c=asf145.get_surface_apex(x_c)+d, y0=y0, R=R50['radius']) #воздух - линза (Be)

asf146 = Surface(lens_aperture = R50['lens aperture']/2, material_nr = material_n, x_c = bsf145.get_surface_apex(R50['lens aperture']/2) + (bsf145.get_surface_apex(R50['lens aperture']/2) - bsf145.get_surface_apex(x_c)) + dist_crl, y0=y0, R=-R50['radius']) #воздух - линза
bsf146 = Surface(lens_aperture = R50['lens aperture']/2, material_nr = 1.0, x_c=asf146.get_surface_apex(x_c)+d, y0=y0, R=R50['radius']) #воздух - линза (Be)

asf147 = Surface(lens_aperture = R50['lens aperture']/2, material_nr = material_n, x_c = bsf146.get_surface_apex(R50['lens aperture']/2) + (bsf146.get_surface_apex(R50['lens aperture']/2) - bsf146.get_surface_apex(x_c)) + dist_crl, y0=y0, R=-R50['radius']) #воздух - линза
bsf147 = Surface(lens_aperture = R50['lens aperture']/2, material_nr = 1.0, x_c=asf147.get_surface_apex(x_c)+d, y0=y0, R=R50['radius']) #воздух - линза (Be)

asf148 = Surface(lens_aperture = R50['lens aperture']/2, material_nr = material_n, x_c = bsf147.get_surface_apex(R50['lens aperture']/2) + (bsf147.get_surface_apex(R50['lens aperture']/2) - bsf147.get_surface_apex(x_c)) + dist_crl, y0=y0, R=-R50['radius']) #воздух - линза
bsf148 = Surface(lens_aperture = R50['lens aperture']/2, material_nr = 1.0, x_c=asf148.get_surface_apex(x_c)+d, y0=y0, R=R50['radius']) #воздух - линза (Be)

asf149 = Surface(lens_aperture = R50['lens aperture']/2, material_nr = material_n, x_c = bsf148.get_surface_apex(R50['lens aperture']/2) + (bsf148.get_surface_apex(R50['lens aperture']/2) - bsf148.get_surface_apex(x_c)) + dist_crl, y0=y0, R=-R50['radius']) #воздух - линза
bsf149 = Surface(lens_aperture = R50['lens aperture']/2, material_nr = 1.0, x_c=asf149.get_surface_apex(x_c)+d, y0=y0, R=R50['radius']) #воздух - линза (Be)

asf150 = Surface(lens_aperture = R50['lens aperture']/2, material_nr = material_n, x_c = bsf149.get_surface_apex(R50['lens aperture']/2) + (bsf149.get_surface_apex(R50['lens aperture']/2) - bsf149.get_surface_apex(x_c)) + dist_crl, y0=y0, R=-R50['radius']) #воздух - линза
bsf150 = Surface(lens_aperture = R50['lens aperture']/2, material_nr = 1.0, x_c=asf150.get_surface_apex(x_c)+d, y0=y0, R=R50['radius']) #воздух - линза (Be)

asf151 = Surface(lens_aperture = R50['lens aperture']/2, material_nr = material_n, x_c = bsf150.get_surface_apex(R50['lens aperture']/2) + (bsf150.get_surface_apex(R50['lens aperture']/2) - bsf150.get_surface_apex(x_c)) + dist_crl, y0=y0, R=-R50['radius']) #воздух - линза
bsf151 = Surface(lens_aperture = R50['lens aperture']/2, material_nr = 1.0, x_c=asf151.get_surface_apex(x_c)+d, y0=y0, R=R50['radius']) #воздух - линза (Be)

asf152 = Surface(lens_aperture = R50['lens aperture']/2, material_nr = material_n, x_c = bsf151.get_surface_apex(R50['lens aperture']/2) + (bsf151.get_surface_apex(R50['lens aperture']/2) - bsf151.get_surface_apex(x_c)) + dist_crl, y0=y0, R=-R50['radius']) #воздух - линза
bsf152 = Surface(lens_aperture = R50['lens aperture']/2, material_nr = 1.0, x_c=asf152.get_surface_apex(x_c)+d, y0=y0, R=R50['radius']) #воздух - линза (Be)

asf153 = Surface(lens_aperture = R50['lens aperture']/2, material_nr = material_n, x_c = bsf152.get_surface_apex(R50['lens aperture']/2) + (bsf152.get_surface_apex(R50['lens aperture']/2) - bsf152.get_surface_apex(x_c)) + dist_crl, y0=y0, R=-R50['radius']) #воздух - линза
bsf153 = Surface(lens_aperture = R50['lens aperture']/2, material_nr = 1.0, x_c=asf153.get_surface_apex(x_c)+d, y0=y0, R=R50['radius']) #воздух - линза (Be)

asf154 = Surface(lens_aperture = R50['lens aperture']/2, material_nr = material_n, x_c = bsf153.get_surface_apex(R50['lens aperture']/2) + (bsf153.get_surface_apex(R50['lens aperture']/2) - bsf153.get_surface_apex(x_c)) + dist_crl, y0=y0, R=-R50['radius']) #воздух - линза
bsf154 = Surface(lens_aperture = R50['lens aperture']/2, material_nr = 1.0, x_c=asf154.get_surface_apex(x_c)+d, y0=y0, R=R50['radius']) #воздух - линза (Be)

asf155 = Surface(lens_aperture = R50['lens aperture']/2, material_nr = material_n, x_c = bsf154.get_surface_apex(R50['lens aperture']/2) + (bsf154.get_surface_apex(R50['lens aperture']/2) - bsf154.get_surface_apex(x_c)) + dist_crl, y0=y0, R=-R50['radius']) #воздух - линза
bsf155 = Surface(lens_aperture = R50['lens aperture']/2, material_nr = 1.0, x_c=asf155.get_surface_apex(x_c)+d, y0=y0, R=R50['radius']) #воздух - линза (Be)

asf156 = Surface(lens_aperture = R50['lens aperture']/2, material_nr = material_n, x_c = bsf155.get_surface_apex(R50['lens aperture']/2) + (bsf155.get_surface_apex(R50['lens aperture']/2) - bsf155.get_surface_apex(x_c)) + dist_crl, y0=y0, R=-R50['radius']) #воздух - линза
bsf156 = Surface(lens_aperture = R50['lens aperture']/2, material_nr = 1.0, x_c=asf156.get_surface_apex(x_c)+d, y0=y0, R=R50['radius']) #воздух - линза (Be)

asf157 = Surface(lens_aperture = R50['lens aperture']/2, material_nr = material_n, x_c = bsf156.get_surface_apex(R50['lens aperture']/2) + (bsf156.get_surface_apex(R50['lens aperture']/2) - bsf156.get_surface_apex(x_c)) + dist_crl, y0=y0, R=-R50['radius']) #воздух - линза
bsf157 = Surface(lens_aperture = R50['lens aperture']/2, material_nr = 1.0, x_c=asf157.get_surface_apex(x_c)+d, y0=y0, R=R50['radius']) #воздух - линза (Be)

asf158 = Surface(lens_aperture = R50['lens aperture']/2, material_nr = material_n, x_c = bsf157.get_surface_apex(R50['lens aperture']/2) + (bsf157.get_surface_apex(R50['lens aperture']/2) - bsf157.get_surface_apex(x_c)) + dist_crl, y0=y0, R=-R50['radius']) #воздух - линза
bsf158 = Surface(lens_aperture = R50['lens aperture']/2, material_nr = 1.0, x_c=asf158.get_surface_apex(x_c)+d, y0=y0, R=R50['radius']) #воздух - линза (Be)

asf159 = Surface(lens_aperture = R50['lens aperture']/2, material_nr = material_n, x_c = bsf158.get_surface_apex(R50['lens aperture']/2) + (bsf158.get_surface_apex(R50['lens aperture']/2) - bsf158.get_surface_apex(x_c)) + dist_crl, y0=y0, R=-R50['radius']) #воздух - линза
bsf159 = Surface(lens_aperture = R50['lens aperture']/2, material_nr = 1.0, x_c=asf159.get_surface_apex(x_c)+d, y0=y0, R=R50['radius']) #воздух - линза (Be)

asf160 = Surface(lens_aperture = R50['lens aperture']/2, material_nr = material_n, x_c = bsf159.get_surface_apex(R50['lens aperture']/2) + (bsf159.get_surface_apex(R50['lens aperture']/2) - bsf159.get_surface_apex(x_c)) + dist_crl, y0=y0, R=-R50['radius']) #воздух - линза
bsf160 = Surface(lens_aperture = R50['lens aperture']/2, material_nr = 1.0, x_c=asf160.get_surface_apex(x_c)+d, y0=y0, R=R50['radius']) #воздух - линза (Be)

asf161 = Surface(lens_aperture = R50['lens aperture']/2, material_nr = material_n, x_c = bsf160.get_surface_apex(R50['lens aperture']/2) + (bsf160.get_surface_apex(R50['lens aperture']/2) - bsf160.get_surface_apex(x_c)) + dist_crl, y0=y0, R=-R50['radius']) #воздух - линза
bsf161 = Surface(lens_aperture = R50['lens aperture']/2, material_nr = 1.0, x_c=asf161.get_surface_apex(x_c)+d, y0=y0, R=R50['radius']) #воздух - линза (Be)

asf162 = Surface(lens_aperture = R50['lens aperture']/2, material_nr = material_n, x_c = bsf161.get_surface_apex(R50['lens aperture']/2) + (bsf161.get_surface_apex(R50['lens aperture']/2) - bsf161.get_surface_apex(x_c)) + dist_crl, y0=y0, R=-R50['radius']) #воздух - линза
bsf162 = Surface(lens_aperture = R50['lens aperture']/2, material_nr = 1.0, x_c=asf162.get_surface_apex(x_c)+d, y0=y0, R=R50['radius']) #воздух - линза (Be)

asf163 = Surface(lens_aperture = R50['lens aperture']/2, material_nr = material_n, x_c = bsf162.get_surface_apex(R50['lens aperture']/2) + (bsf162.get_surface_apex(R50['lens aperture']/2) - bsf162.get_surface_apex(x_c)) + dist_crl, y0=y0, R=-R50['radius']) #воздух - линза
bsf163 = Surface(lens_aperture = R50['lens aperture']/2, material_nr = 1.0, x_c=asf163.get_surface_apex(x_c)+d, y0=y0, R=R50['radius']) #воздух - линза (Be)

asf164 = Surface(lens_aperture = R50['lens aperture']/2, material_nr = material_n, x_c = bsf163.get_surface_apex(R50['lens aperture']/2) + (bsf163.get_surface_apex(R50['lens aperture']/2) - bsf163.get_surface_apex(x_c)) + dist_crl, y0=y0, R=-R50['radius']) #воздух - линза
bsf164 = Surface(lens_aperture = R50['lens aperture']/2, material_nr = 1.0, x_c=asf164.get_surface_apex(x_c)+d, y0=y0, R=R50['radius']) #воздух - линза (Be)

asf165 = Surface(lens_aperture = R50['lens aperture']/2, material_nr = material_n, x_c = bsf164.get_surface_apex(R50['lens aperture']/2) + (bsf164.get_surface_apex(R50['lens aperture']/2) - bsf164.get_surface_apex(x_c)) + dist_crl, y0=y0, R=-R50['radius']) #воздух - линза
bsf165 = Surface(lens_aperture = R50['lens aperture']/2, material_nr = 1.0, x_c=asf165.get_surface_apex(x_c)+d, y0=y0, R=R50['radius']) #воздух - линза (Be)

asf166 = Surface(lens_aperture = R50['lens aperture']/2, material_nr = material_n, x_c = bsf165.get_surface_apex(R50['lens aperture']/2) + (bsf165.get_surface_apex(R50['lens aperture']/2) - bsf165.get_surface_apex(x_c)) + dist_crl, y0=y0, R=-R50['radius']) #воздух - линза
bsf166 = Surface(lens_aperture = R50['lens aperture']/2, material_nr = 1.0, x_c=asf166.get_surface_apex(x_c)+d, y0=y0, R=R50['radius']) #воздух - линза (Be)

asf167 = Surface(lens_aperture = R50['lens aperture']/2, material_nr = material_n, x_c = bsf166.get_surface_apex(R50['lens aperture']/2) + (bsf166.get_surface_apex(R50['lens aperture']/2) - bsf166.get_surface_apex(x_c)) + dist_crl, y0=y0, R=-R50['radius']) #воздух - линза
bsf167 = Surface(lens_aperture = R50['lens aperture']/2, material_nr = 1.0, x_c=asf167.get_surface_apex(x_c)+d, y0=y0, R=R50['radius']) #воздух - линза (Be)

asf168 = Surface(lens_aperture = R50['lens aperture']/2, material_nr = material_n, x_c = bsf167.get_surface_apex(R50['lens aperture']/2) + (bsf167.get_surface_apex(R50['lens aperture']/2) - bsf167.get_surface_apex(x_c)) + dist_crl, y0=y0, R=-R50['radius']) #воздух - линза
bsf168 = Surface(lens_aperture = R50['lens aperture']/2, material_nr = 1.0, x_c=asf168.get_surface_apex(x_c)+d, y0=y0, R=R50['radius']) #воздух - линза (Be)

asf169 = Surface(lens_aperture = R50['lens aperture']/2, material_nr = material_n, x_c = bsf168.get_surface_apex(R50['lens aperture']/2) + (bsf168.get_surface_apex(R50['lens aperture']/2) - bsf168.get_surface_apex(x_c)) + dist_crl, y0=y0, R=-R50['radius']) #воздух - линза
bsf169 = Surface(lens_aperture = R50['lens aperture']/2, material_nr = 1.0, x_c=asf169.get_surface_apex(x_c)+d, y0=y0, R=R50['radius']) #воздух - линза (Be)

asf170 = Surface(lens_aperture = R50['lens aperture']/2, material_nr = material_n, x_c = bsf169.get_surface_apex(R50['lens aperture']/2) + (bsf169.get_surface_apex(R50['lens aperture']/2) - bsf169.get_surface_apex(x_c)) + dist_crl, y0=y0, R=-R50['radius']) #воздух - линза
bsf170 = Surface(lens_aperture = R50['lens aperture']/2, material_nr = 1.0, x_c=asf170.get_surface_apex(x_c)+d, y0=y0, R=R50['radius']) #воздух - линза (Be)

asf171 = Surface(lens_aperture = R50['lens aperture']/2, material_nr = material_n, x_c = bsf170.get_surface_apex(R50['lens aperture']/2) + (bsf170.get_surface_apex(R50['lens aperture']/2) - bsf170.get_surface_apex(x_c)) + dist_crl, y0=y0, R=-R50['radius']) #воздух - линза
bsf171 = Surface(lens_aperture = R50['lens aperture']/2, material_nr = 1.0, x_c=asf171.get_surface_apex(x_c)+d, y0=y0, R=R50['radius']) #воздух - линза (Be)

asf172 = Surface(lens_aperture = R50['lens aperture']/2, material_nr = material_n, x_c = bsf171.get_surface_apex(R50['lens aperture']/2) + (bsf171.get_surface_apex(R50['lens aperture']/2) - bsf171.get_surface_apex(x_c)) + dist_crl, y0=y0, R=-R50['radius']) #воздух - линза
bsf172 = Surface(lens_aperture = R50['lens aperture']/2, material_nr = 1.0, x_c=asf172.get_surface_apex(x_c)+d, y0=y0, R=R50['radius']) #воздух - линза (Be)

asf173 = Surface(lens_aperture = R50['lens aperture']/2, material_nr = material_n, x_c = bsf172.get_surface_apex(R50['lens aperture']/2) + (bsf172.get_surface_apex(R50['lens aperture']/2) - bsf172.get_surface_apex(x_c)) + dist_crl, y0=y0, R=-R50['radius']) #воздух - линза
bsf173 = Surface(lens_aperture = R50['lens aperture']/2, material_nr = 1.0, x_c=asf173.get_surface_apex(x_c)+d, y0=y0, R=R50['radius']) #воздух - линза (Be)

asf174 = Surface(lens_aperture = R50['lens aperture']/2, material_nr = material_n, x_c = bsf173.get_surface_apex(R50['lens aperture']/2) + (bsf173.get_surface_apex(R50['lens aperture']/2) - bsf173.get_surface_apex(x_c)) + dist_crl, y0=y0, R=-R50['radius']) #воздух - линза
bsf174 = Surface(lens_aperture = R50['lens aperture']/2, material_nr = 1.0, x_c=asf174.get_surface_apex(x_c)+d, y0=y0, R=R50['radius']) #воздух - линза (Be)

asf175 = Surface(lens_aperture = R50['lens aperture']/2, material_nr = material_n, x_c = bsf174.get_surface_apex(R50['lens aperture']/2) + (bsf174.get_surface_apex(R50['lens aperture']/2) - bsf174.get_surface_apex(x_c)) + dist_crl, y0=y0, R=-R50['radius']) #воздух - линза
bsf175 = Surface(lens_aperture = R50['lens aperture']/2, material_nr = 1.0, x_c=asf175.get_surface_apex(x_c)+d, y0=y0, R=R50['radius']) #воздух - линза (Be)

asf176 = Surface(lens_aperture = R50['lens aperture']/2, material_nr = material_n, x_c = bsf175.get_surface_apex(R50['lens aperture']/2) + (bsf175.get_surface_apex(R50['lens aperture']/2) - bsf175.get_surface_apex(x_c)) + dist_crl, y0=y0, R=-R50['radius']) #воздух - линза
bsf176 = Surface(lens_aperture = R50['lens aperture']/2, material_nr = 1.0, x_c=asf176.get_surface_apex(x_c)+d, y0=y0, R=R50['radius']) #воздух - линза (Be)

asf177 = Surface(lens_aperture = R50['lens aperture']/2, material_nr = material_n, x_c = bsf176.get_surface_apex(R50['lens aperture']/2) + (bsf176.get_surface_apex(R50['lens aperture']/2) - bsf176.get_surface_apex(x_c)) + dist_crl, y0=y0, R=-R50['radius']) #воздух - линза
bsf177 = Surface(lens_aperture = R50['lens aperture']/2, material_nr = 1.0, x_c=asf177.get_surface_apex(x_c)+d, y0=y0, R=R50['radius']) #воздух - линза (Be)

asf178 = Surface(lens_aperture = R50['lens aperture']/2, material_nr = material_n, x_c = bsf177.get_surface_apex(R50['lens aperture']/2) + (bsf177.get_surface_apex(R50['lens aperture']/2) - bsf177.get_surface_apex(x_c)) + dist_crl, y0=y0, R=-R50['radius']) #воздух - линза
bsf178 = Surface(lens_aperture = R50['lens aperture']/2, material_nr = 1.0, x_c=asf178.get_surface_apex(x_c)+d, y0=y0, R=R50['radius']) #воздух - линза (Be)

asf179 = Surface(lens_aperture = R50['lens aperture']/2, material_nr = material_n, x_c = bsf178.get_surface_apex(R50['lens aperture']/2) + (bsf178.get_surface_apex(R50['lens aperture']/2) - bsf178.get_surface_apex(x_c)) + dist_crl, y0=y0, R=-R50['radius']) #воздух - линза
bsf179 = Surface(lens_aperture = R50['lens aperture']/2, material_nr = 1.0, x_c=asf179.get_surface_apex(x_c)+d, y0=y0, R=R50['radius']) #воздух - линза (Be)

asf180 = Surface(lens_aperture = R50['lens aperture']/2, material_nr = material_n, x_c = bsf179.get_surface_apex(R50['lens aperture']/2) + (bsf179.get_surface_apex(R50['lens aperture']/2) - bsf179.get_surface_apex(x_c)) + dist_crl, y0=y0, R=-R50['radius']) #воздух - линза
bsf180 = Surface(lens_aperture = R50['lens aperture']/2, material_nr = 1.0, x_c=asf180.get_surface_apex(x_c)+d, y0=y0, R=R50['radius']) #воздух - линза (Be)

asf181 = Surface(lens_aperture = R50['lens aperture']/2, material_nr = material_n, x_c = bsf180.get_surface_apex(R50['lens aperture']/2) + (bsf180.get_surface_apex(R50['lens aperture']/2) - bsf180.get_surface_apex(x_c)) + dist_crl, y0=y0, R=-R50['radius']) #воздух - линза
bsf181 = Surface(lens_aperture = R50['lens aperture']/2, material_nr = 1.0, x_c=asf181.get_surface_apex(x_c)+d, y0=y0, R=R50['radius']) #воздух - линза (Be)

asf182 = Surface(lens_aperture = R50['lens aperture']/2, material_nr = material_n, x_c = bsf181.get_surface_apex(R50['lens aperture']/2) + (bsf181.get_surface_apex(R50['lens aperture']/2) - bsf181.get_surface_apex(x_c)) + dist_crl, y0=y0, R=-R50['radius']) #воздух - линза
bsf182 = Surface(lens_aperture = R50['lens aperture']/2, material_nr = 1.0, x_c=asf182.get_surface_apex(x_c)+d, y0=y0, R=R50['radius']) #воздух - линза (Be)

asf183 = Surface(lens_aperture = R50['lens aperture']/2, material_nr = material_n, x_c = bsf182.get_surface_apex(R50['lens aperture']/2) + (bsf182.get_surface_apex(R50['lens aperture']/2) - bsf182.get_surface_apex(x_c)) + dist_crl, y0=y0, R=-R50['radius']) #воздух - линза
bsf183 = Surface(lens_aperture = R50['lens aperture']/2, material_nr = 1.0, x_c=asf183.get_surface_apex(x_c)+d, y0=y0, R=R50['radius']) #воздух - линза (Be)

asf184 = Surface(lens_aperture = R50['lens aperture']/2, material_nr = material_n, x_c = bsf183.get_surface_apex(R50['lens aperture']/2) + (bsf183.get_surface_apex(R50['lens aperture']/2) - bsf183.get_surface_apex(x_c)) + dist_crl, y0=y0, R=-R50['radius']) #воздух - линза
bsf184 = Surface(lens_aperture = R50['lens aperture']/2, material_nr = 1.0, x_c=asf184.get_surface_apex(x_c)+d, y0=y0, R=R50['radius']) #воздух - линза (Be)

asf185 = Surface(lens_aperture = R50['lens aperture']/2, material_nr = material_n, x_c = bsf184.get_surface_apex(R50['lens aperture']/2) + (bsf184.get_surface_apex(R50['lens aperture']/2) - bsf184.get_surface_apex(x_c)) + dist_crl, y0=y0, R=-R50['radius']) #воздух - линза
bsf185 = Surface(lens_aperture = R50['lens aperture']/2, material_nr = 1.0, x_c=asf185.get_surface_apex(x_c)+d, y0=y0, R=R50['radius']) #воздух - линза (Be)

asf186 = Surface(lens_aperture = R50['lens aperture']/2, material_nr = material_n, x_c = bsf185.get_surface_apex(R50['lens aperture']/2) + (bsf185.get_surface_apex(R50['lens aperture']/2) - bsf185.get_surface_apex(x_c)) + dist_crl, y0=y0, R=-R50['radius']) #воздух - линза
bsf186 = Surface(lens_aperture = R50['lens aperture']/2, material_nr = 1.0, x_c=asf186.get_surface_apex(x_c)+d, y0=y0, R=R50['radius']) #воздух - линза (Be)

asf187 = Surface(lens_aperture = R50['lens aperture']/2, material_nr = material_n, x_c = bsf186.get_surface_apex(R50['lens aperture']/2) + (bsf186.get_surface_apex(R50['lens aperture']/2) - bsf186.get_surface_apex(x_c)) + dist_crl, y0=y0, R=-R50['radius']) #воздух - линза
bsf187 = Surface(lens_aperture = R50['lens aperture']/2, material_nr = 1.0, x_c=asf187.get_surface_apex(x_c)+d, y0=y0, R=R50['radius']) #воздух - линза (Be)

asf188 = Surface(lens_aperture = R50['lens aperture']/2, material_nr = material_n, x_c = bsf187.get_surface_apex(R50['lens aperture']/2) + (bsf187.get_surface_apex(R50['lens aperture']/2) - bsf187.get_surface_apex(x_c)) + dist_crl, y0=y0, R=-R50['radius']) #воздух - линза
bsf188 = Surface(lens_aperture = R50['lens aperture']/2, material_nr = 1.0, x_c=asf188.get_surface_apex(x_c)+d, y0=y0, R=R50['radius']) #воздух - линза (Be)

asf189 = Surface(lens_aperture = R50['lens aperture']/2, material_nr = material_n, x_c = bsf188.get_surface_apex(R50['lens aperture']/2) + (bsf188.get_surface_apex(R50['lens aperture']/2) - bsf188.get_surface_apex(x_c)) + dist_crl, y0=y0, R=-R50['radius']) #воздух - линза
bsf189 = Surface(lens_aperture = R50['lens aperture']/2, material_nr = 1.0, x_c=asf189.get_surface_apex(x_c)+d, y0=y0, R=R50['radius']) #воздух - линза (Be)

asf190 = Surface(lens_aperture = R50['lens aperture']/2, material_nr = material_n, x_c = bsf189.get_surface_apex(R50['lens aperture']/2) + (bsf189.get_surface_apex(R50['lens aperture']/2) - bsf189.get_surface_apex(x_c)) + dist_crl, y0=y0, R=-R50['radius']) #воздух - линза
bsf190 = Surface(lens_aperture = R50['lens aperture']/2, material_nr = 1.0, x_c=asf190.get_surface_apex(x_c)+d, y0=y0, R=R50['radius']) #воздух - линза (Be)

asf191 = Surface(lens_aperture = R50['lens aperture']/2, material_nr = material_n, x_c = bsf190.get_surface_apex(R50['lens aperture']/2) + (bsf190.get_surface_apex(R50['lens aperture']/2) - bsf190.get_surface_apex(x_c)) + dist_crl, y0=y0, R=-R50['radius']) #воздух - линза
bsf191 = Surface(lens_aperture = R50['lens aperture']/2, material_nr = 1.0, x_c=asf191.get_surface_apex(x_c)+d, y0=y0, R=R50['radius']) #воздух - линза (Be)

asf192 = Surface(lens_aperture = R50['lens aperture']/2, material_nr = material_n, x_c = bsf191.get_surface_apex(R50['lens aperture']/2) + (bsf191.get_surface_apex(R50['lens aperture']/2) - bsf191.get_surface_apex(x_c)) + dist_crl, y0=y0, R=-R50['radius']) #воздух - линза
bsf192 = Surface(lens_aperture = R50['lens aperture']/2, material_nr = 1.0, x_c=asf192.get_surface_apex(x_c)+d, y0=y0, R=R50['radius']) #воздух - линза (Be)

asf193 = Surface(lens_aperture = R50['lens aperture']/2, material_nr = material_n, x_c = bsf192.get_surface_apex(R50['lens aperture']/2) + (bsf192.get_surface_apex(R50['lens aperture']/2) - bsf192.get_surface_apex(x_c)) + dist_crl, y0=y0, R=-R50['radius']) #воздух - линза
bsf193 = Surface(lens_aperture = R50['lens aperture']/2, material_nr = 1.0, x_c=asf193.get_surface_apex(x_c)+d, y0=y0, R=R50['radius']) #воздух - линза (Be)

asf194 = Surface(lens_aperture = R50['lens aperture']/2, material_nr = material_n, x_c = bsf193.get_surface_apex(R50['lens aperture']/2) + (bsf193.get_surface_apex(R50['lens aperture']/2) - bsf193.get_surface_apex(x_c)) + dist_crl, y0=y0, R=-R50['radius']) #воздух - линза
bsf194 = Surface(lens_aperture = R50['lens aperture']/2, material_nr = 1.0, x_c=asf194.get_surface_apex(x_c)+d, y0=y0, R=R50['radius']) #воздух - линза (Be)

asf195 = Surface(lens_aperture = R50['lens aperture']/2, material_nr = material_n, x_c = bsf194.get_surface_apex(R50['lens aperture']/2) + (bsf194.get_surface_apex(R50['lens aperture']/2) - bsf194.get_surface_apex(x_c)) + dist_crl, y0=y0, R=-R50['radius']) #воздух - линза
bsf195 = Surface(lens_aperture = R50['lens aperture']/2, material_nr = 1.0, x_c=asf195.get_surface_apex(x_c)+d, y0=y0, R=R50['radius']) #воздух - линза (Be)

asf196 = Surface(lens_aperture = R50['lens aperture']/2, material_nr = material_n, x_c = bsf195.get_surface_apex(R50['lens aperture']/2) + (bsf195.get_surface_apex(R50['lens aperture']/2) - bsf195.get_surface_apex(x_c)) + dist_crl, y0=y0, R=-R50['radius']) #воздух - линза
bsf196 = Surface(lens_aperture = R50['lens aperture']/2, material_nr = 1.0, x_c=asf196.get_surface_apex(x_c)+d, y0=y0, R=R50['radius']) #воздух - линза (Be)

asf197 = Surface(lens_aperture = R50['lens aperture']/2, material_nr = material_n, x_c = bsf196.get_surface_apex(R50['lens aperture']/2) + (bsf196.get_surface_apex(R50['lens aperture']/2) - bsf196.get_surface_apex(x_c)) + dist_crl, y0=y0, R=-R50['radius']) #воздух - линза
bsf197 = Surface(lens_aperture = R50['lens aperture']/2, material_nr = 1.0, x_c=asf197.get_surface_apex(x_c)+d, y0=y0, R=R50['radius']) #воздух - линза (Be)

asf198 = Surface(lens_aperture = R50['lens aperture']/2, material_nr = material_n, x_c = bsf197.get_surface_apex(R50['lens aperture']/2) + (bsf197.get_surface_apex(R50['lens aperture']/2) - bsf197.get_surface_apex(x_c)) + dist_crl, y0=y0, R=-R50['radius']) #воздух - линза
bsf198 = Surface(lens_aperture = R50['lens aperture']/2, material_nr = 1.0, x_c=asf198.get_surface_apex(x_c)+d, y0=y0, R=R50['radius']) #воздух - линза (Be)

asf199 = Surface(lens_aperture = R50['lens aperture']/2, material_nr = material_n, x_c = bsf198.get_surface_apex(R50['lens aperture']/2) + (bsf198.get_surface_apex(R50['lens aperture']/2) - bsf198.get_surface_apex(x_c)) + dist_crl, y0=y0, R=-R50['radius']) #воздух - линза
bsf199 = Surface(lens_aperture = R50['lens aperture']/2, material_nr = 1.0, x_c=asf199.get_surface_apex(x_c)+d, y0=y0, R=R50['radius']) #воздух - линза (Be)

asf200 = Surface(lens_aperture = R50['lens aperture']/2, material_nr = material_n, x_c = bsf199.get_surface_apex(R50['lens aperture']/2) + (bsf199.get_surface_apex(R50['lens aperture']/2) - bsf199.get_surface_apex(x_c)) + dist_crl, y0=y0, R=-R50['radius']) #воздух - линза
bsf200 = Surface(lens_aperture = R50['lens aperture']/2, material_nr = 1.0, x_c=asf200.get_surface_apex(x_c)+d, y0=y0, R=R50['radius']) #воздух - линза (Be)

asf201 = Surface(lens_aperture = R50['lens aperture']/2, material_nr = material_n, x_c = bsf200.get_surface_apex(R50['lens aperture']/2) + (bsf200.get_surface_apex(R50['lens aperture']/2) - bsf200.get_surface_apex(x_c)) + dist_crl, y0=y0, R=-R50['radius']) #воздух - линза
bsf201 = Surface(lens_aperture = R50['lens aperture']/2, material_nr = 1.0, x_c=asf201.get_surface_apex(x_c)+d, y0=y0, R=R50['radius']) #воздух - линза (Be)`

asf202 = Surface(lens_aperture = R50['lens aperture']/2, material_nr = material_n, x_c = bsf201.get_surface_apex(R50['lens aperture']/2) + (bsf201.get_surface_apex(R50['lens aperture']/2) - bsf201.get_surface_apex(x_c)) + dist_crl, y0=y0, R=-R50['radius']) #воздух - линза
bsf202 = Surface(lens_aperture = R50['lens aperture']/2, material_nr = 1.0, x_c=asf202.get_surface_apex(x_c)+d, y0=y0, R=R50['radius']) #воздух - линза (Be)

asf203 = Surface(lens_aperture = R50['lens aperture']/2, material_nr = material_n, x_c = bsf202.get_surface_apex(R50['lens aperture']/2) + (bsf202.get_surface_apex(R50['lens aperture']/2) - bsf202.get_surface_apex(x_c)) + dist_crl, y0=y0, R=-R50['radius']) #воздух - линза
bsf203 = Surface(lens_aperture = R50['lens aperture']/2, material_nr = 1.0, x_c=asf203.get_surface_apex(x_c)+d, y0=y0, R=R50['radius']) #воздух - линза (Be)

asf204 = Surface(lens_aperture = R50['lens aperture']/2, material_nr = material_n, x_c = bsf203.get_surface_apex(R50['lens aperture']/2) + (bsf203.get_surface_apex(R50['lens aperture']/2) - bsf203.get_surface_apex(x_c)) + dist_crl, y0=y0, R=-R50['radius']) #воздух - линза
bsf204 = Surface(lens_aperture = R50['lens aperture']/2, material_nr = 1.0, x_c=asf204.get_surface_apex(x_c)+d, y0=y0, R=R50['radius']) #воздух - линза (Be)

asf205 = Surface(lens_aperture = R50['lens aperture']/2, material_nr = material_n, x_c = bsf204.get_surface_apex(R50['lens aperture']/2) + (bsf204.get_surface_apex(R50['lens aperture']/2) - bsf204.get_surface_apex(x_c)) + dist_crl, y0=y0, R=-R50['radius']) #воздух - линза
bsf205 = Surface(lens_aperture = R50['lens aperture']/2, material_nr = 1.0, x_c=asf205.get_surface_apex(x_c)+d, y0=y0, R=R50['radius']) #воздух - линза (Be)

asf206 = Surface(lens_aperture = R50['lens aperture']/2, material_nr = material_n, x_c = bsf205.get_surface_apex(R50['lens aperture']/2) + (bsf205.get_surface_apex(R50['lens aperture']/2) - bsf205.get_surface_apex(x_c)) + dist_crl, y0=y0, R=-R50['radius']) #воздух - линза
bsf206 = Surface(lens_aperture = R50['lens aperture']/2, material_nr = 1.0, x_c=asf206.get_surface_apex(x_c)+d, y0=y0, R=R50['radius']) #воздух - линза (Be)

asf207 = Surface(lens_aperture = R50['lens aperture']/2, material_nr = material_n, x_c = bsf206.get_surface_apex(R50['lens aperture']/2) + (bsf206.get_surface_apex(R50['lens aperture']/2) - bsf206.get_surface_apex(x_c)) + dist_crl, y0=y0, R=-R50['radius']) #воздух - линза
bsf207 = Surface(lens_aperture = R50['lens aperture']/2, material_nr = 1.0, x_c=asf207.get_surface_apex(x_c)+d, y0=y0, R=R50['radius']) #воздух - линза (Be)

asf208 = Surface(lens_aperture = R50['lens aperture']/2, material_nr = material_n, x_c = bsf207.get_surface_apex(R50['lens aperture']/2) + (bsf207.get_surface_apex(R50['lens aperture']/2) - bsf207.get_surface_apex(x_c)) + dist_crl, y0=y0, R=-R50['radius']) #воздух - линза
bsf208 = Surface(lens_aperture = R50['lens aperture']/2, material_nr = 1.0, x_c=asf208.get_surface_apex(x_c)+d, y0=y0, R=R50['radius']) #воздух - линза (Be)

#Building interaction and visualization

#plt.rcParams["figure.figsize"] = 7,7
#ax1 = plt.subplot2grid

#fig, ax1, ax2 = plt.figure(figsize=(12,4), tight_layout = True), plt.subplots(2,1), plt.subplots(1,2)
fig = plt.figure(figsize=(8, 5))
ax1 = plt.subplot2grid((3,4), (0, 0), colspan = 4, rowspan = 2)
#ax1 = plt.subplots(1,2)
ax1.grid()
#fig.set_figheight(8)
#fig.set_figwidth(20)

#ax.set_xlabel('x, m')
#ax.xaxis.set_major_locator(plt.MultipleLocator(1e6))
'''
Сделать инциализацию трансфокаторов, кол-во и позиции введённых линз через словари

'''

for i in range(y_start.shape[0]):
    for j in range(theta.shape[0]):
        ray_wavelength = 0.1
        ray = Ray(x_start[i], y_start[i], theta[j], ray_wavelength)
        rays.append(ray)
        '''
        asf1.intersection(ray, t_min=0, t_max=50, prev_n=None)
        bsf1.intersection(ray, t_min=0, t_max=50, prev_n=asf1.n_r)
        asf1.render(ax1)
        bsf1.render(ax1)
        
        asf2.intersection(ray, t_min=0, t_max=50, prev_n=None)
        bsf2.intersection(ray, t_min=0, t_max=50, prev_n=asf2.n_r)
        
        asf3.intersection(ray, t_min=0, t_max=50, prev_n=None)
        bsf3.intersection(ray, t_min=0, t_max=50, prev_n=asf3.n_r)

        asf4.intersection(ray, t_min=0, t_max=50, prev_n=None)
        bsf4.intersection(ray, t_min=0, t_max=50, prev_n=asf4.n_r)

        asf5.intersection(ray, t_min=0, t_max=50, prev_n=None)
        bsf5.intersection(ray, t_min=0, t_max=50, prev_n=asf5.n_r)
        
        asf6.intersection(ray, t_min=0, t_max=50, prev_n=None)
        bsf6.intersection(ray, t_min=0, t_max=50, prev_n=asf6.n_r)

        asf7.intersection(ray, t_min=0, t_max=50, prev_n=None)
        bsf7.intersection(ray, t_min=0, t_max=50, prev_n=asf7.n_r)

        asf8.intersection(ray, t_min=0, t_max=50, prev_n=None)
        bsf8.intersection(ray, t_min=0, t_max=50, prev_n=asf8.n_r)

        asf9.intersection(ray, t_min=0, t_max=50, prev_n=None)
        bsf9.intersection(ray, t_min=0, t_max=50, prev_n=asf9.n_r)       

        asf10.intersection(ray, t_min=0, t_max=50, prev_n=None)
        bsf10.intersection(ray, t_min=0, t_max=50, prev_n=asf10.n_r)

        asf11.intersection(ray, t_min=0, t_max=50, prev_n=None)
        bsf11.intersection(ray, t_min=0, t_max=50, prev_n=asf11.n_r)

        asf12.intersection(ray, t_min=0, t_max=50, prev_n=None)
        bsf12.intersection(ray, t_min=0, t_max=50, prev_n=asf12.n_r)
        
        asf13.intersection(ray, t_min=0, t_max=50, prev_n=None)
        bsf13.intersection(ray, t_min=0, t_max=50, prev_n=asf13.n_r)
        
        asf14.intersection(ray, t_min=0, t_max=50, prev_n=None)
        bsf14.intersection(ray, t_min=0, t_max=50, prev_n=asf14.n_r)
        
        asf15.intersection(ray, t_min=0, t_max=50, prev_n=None)
        bsf15.intersection(ray, t_min=0, t_max=50, prev_n=asf15.n_r)
        
        asf16.intersection(ray, t_min=0, t_max=50, prev_n=None)
        bsf16.intersection(ray, t_min=0, t_max=50, prev_n=asf16.n_r)
        
        asf17.intersection(ray, t_min=0, t_max=50, prev_n=None)
        bsf17.intersection(ray, t_min=0, t_max=50, prev_n=asf17.n_r)
        
        asf18.intersection(ray, t_min=0, t_max=50, prev_n=None)
        bsf18.intersection(ray, t_min=0, t_max=50, prev_n=asf18.n_r)
        
        asf19.intersection(ray, t_min=0, t_max=50, prev_n=None)
        bsf19.intersection(ray, t_min=0, t_max=50, prev_n=asf19.n_r)
        
        asf20.intersection(ray, t_min=0, t_max=50, prev_n=None)
        bsf20.intersection(ray, t_min=0, t_max=50, prev_n=asf20.n_r)
        
        asf21.intersection(ray, t_min=0, t_max=50, prev_n=None)
        bsf21.intersection(ray, t_min=0, t_max=50, prev_n=asf21.n_r)
        
        asf22.intersection(ray, t_min=0, t_max=50, prev_n=None)
        bsf22.intersection(ray, t_min=0, t_max=50, prev_n=asf22.n_r)
        
        asf23.intersection(ray, t_min=0, t_max=50, prev_n=None)
        bsf23.intersection(ray, t_min=0, t_max=50, prev_n=asf23.n_r)
        
        asf24.intersection(ray, t_min=0, t_max=50, prev_n=None)
        bsf24.intersection(ray, t_min=0, t_max=50, prev_n=asf24.n_r)
        
        asf25.intersection(ray, t_min=0, t_max=50, prev_n=None)
        bsf25.intersection(ray, t_min=0, t_max=50, prev_n=asf25.n_r)
        
        asf26.intersection(ray, t_min=0, t_max=50, prev_n=None)
        bsf26.intersection(ray, t_min=0, t_max=50, prev_n=asf26.n_r)
        
        asf27.intersection(ray, t_min=0, t_max=50, prev_n=None)
        bsf27.intersection(ray, t_min=0, t_max=50, prev_n=asf27.n_r)
        
        asf28.intersection(ray, t_min=0, t_max=50, prev_n=None)
        bsf28.intersection(ray, t_min=0, t_max=50, prev_n=asf28.n_r)
        
        asf29.intersection(ray, t_min=0, t_max=50, prev_n=None)
        bsf29.intersection(ray, t_min=0, t_max=50, prev_n=asf29.n_r)
        
        asf30.intersection(ray, t_min=0, t_max=50, prev_n=None)
        bsf30.intersection(ray, t_min=0, t_max=50, prev_n=asf30.n_r)
        
        asf31.intersection(ray, t_min=0, t_max=50, prev_n=None)
        bsf31.intersection(ray, t_min=0, t_max=50, prev_n=asf31.n_r)
            
        asf32.intersection(ray, t_min=0, t_max=50, prev_n=None)
        bsf32.intersection(ray, t_min=0, t_max=50, prev_n=asf32.n_r)
            
        asf33.intersection(ray, t_min=0, t_max=50, prev_n=None)
        bsf33.intersection(ray, t_min=0, t_max=50, prev_n=asf33.n_r)

        asf34.intersection(ray, t_min=0, t_max=50, prev_n=None)
        bsf34.intersection(ray, t_min=0, t_max=50, prev_n=asf34.n_r)
            
        asf35.intersection(ray, t_min=0, t_max=50, prev_n=None)
        bsf35.intersection(ray, t_min=0, t_max=50, prev_n=asf35.n_r)
            
        asf36.intersection(ray, t_min=0, t_max=50, prev_n=None)
        bsf36.intersection(ray, t_min=0, t_max=50, prev_n=asf36.n_r)
            
        asf37.intersection(ray, t_min=0, t_max=50, prev_n=None)
        bsf37.intersection(ray, t_min=0, t_max=50, prev_n=asf37.n_r)
            
        asf38.intersection(ray, t_min=0, t_max=50, prev_n=None)
        bsf38.intersection(ray, t_min=0, t_max=50, prev_n=asf38.n_r)
        
        asf39.intersection(ray, t_min=0, t_max=50, prev_n=None)
        bsf39.intersection(ray, t_min=0, t_max=50, prev_n=asf39.n_r)
             
        asf40.intersection(ray, t_min=0, t_max=50, prev_n=None)
        bsf40.intersection(ray, t_min=0, t_max=50, prev_n=asf40.n_r)
                
        asf41.intersection(ray, t_min=0, t_max=50, prev_n=None)
        bsf41.intersection(ray, t_min=0, t_max=50, prev_n=asf41.n_r)
            
        asf42.intersection(ray, t_min=0, t_max=50, prev_n=None)
        bsf42.intersection(ray, t_min=0, t_max=50, prev_n=asf42.n_r)
            
        asf43.intersection(ray, t_min=0, t_max=50, prev_n=None)
        bsf43.intersection(ray, t_min=0, t_max=50, prev_n=asf43.n_r)
           
        asf44.intersection(ray, t_min=0, t_max=50, prev_n=None)
        bsf44.intersection(ray, t_min=0, t_max=50, prev_n=asf44.n_r)
            
        asf45.intersection(ray, t_min=0, t_max=50, prev_n=None)
        bsf45.intersection(ray, t_min=0, t_max=50, prev_n=asf45.n_r)
           
        asf46.intersection(ray, t_min=0, t_max=50, prev_n=None)
        bsf46.intersection(ray, t_min=0, t_max=50, prev_n=asf46.n_r)
            
        asf47.intersection(ray, t_min=0, t_max=50, prev_n=None)
        bsf47.intersection(ray, t_min=0, t_max=50, prev_n=asf47.n_r)
            
        asf48.intersection(ray, t_min=0, t_max=50, prev_n=None)
        bsf48.intersection(ray, t_min=0, t_max=50, prev_n=asf48.n_r)
            
        asf49.intersection(ray, t_min=0, t_max=50, prev_n=None)
        bsf49.intersection(ray, t_min=0, t_max=50, prev_n=asf49.n_r)
            
        asf50.intersection(ray, t_min=0, t_max=50, prev_n=None)
        bsf50.intersection(ray, t_min=0, t_max=50, prev_n=asf50.n_r)
                    
        asf51.intersection(ray, t_min=0, t_max=50, prev_n=None)
        bsf51.intersection(ray, t_min=0, t_max=50, prev_n=asf51.n_r)
            
        asf52.intersection(ray, t_min=0, t_max=50, prev_n=None)
        bsf52.intersection(ray, t_min=0, t_max=50, prev_n=asf52.n_r)
            
        asf53.intersection(ray, t_min=0, t_max=50, prev_n=None)
        bsf53.intersection(ray, t_min=0, t_max=50, prev_n=asf53.n_r)
            
        asf54.intersection(ray, t_min=0, t_max=50, prev_n=None)
        bsf54.intersection(ray, t_min=0, t_max=50, prev_n=asf54.n_r)
            
        asf55.intersection(ray, t_min=0, t_max=50, prev_n=None)
        bsf55.intersection(ray, t_min=0, t_max=50, prev_n=asf55.n_r)
            
        asf56.intersection(ray, t_min=0, t_max=50, prev_n=None)
        bsf56.intersection(ray, t_min=0, t_max=50, prev_n=asf56.n_r)
           
        asf57.intersection(ray, t_min=0, t_max=50, prev_n=None)
        bsf57.intersection(ray, t_min=0, t_max=50, prev_n=asf57.n_r)
           
        asf58.intersection(ray, t_min=0, t_max=50, prev_n=None)
        bsf58.intersection(ray, t_min=0, t_max=50, prev_n=asf58.n_r)
            
        asf59.intersection(ray, t_min=0, t_max=50, prev_n=None)
        bsf59.intersection(ray, t_min=0, t_max=50, prev_n=asf59.n_r)
            
        asf60.intersection(ray, t_min=0, t_max=50, prev_n=None)
        bsf60.intersection(ray, t_min=0, t_max=50, prev_n=asf60.n_r)
                    
        asf61.intersection(ray, t_min=0, t_max=50, prev_n=None)
        bsf61.intersection(ray, t_min=0, t_max=50, prev_n=asf61.n_r)
            
        asf62.intersection(ray, t_min=0, t_max=50, prev_n=None)
        bsf62.intersection(ray, t_min=0, t_max=50, prev_n=asf62.n_r)
            
        asf63.intersection(ray, t_min=0, t_max=50, prev_n=None)
        bsf63.intersection(ray, t_min=0, t_max=50, prev_n=asf63.n_r)
            
        asf64.intersection(ray, t_min=0, t_max=50, prev_n=None)
        bsf64.intersection(ray, t_min=0, t_max=50, prev_n=asf64.n_r)
            
        asf65.intersection(ray, t_min=0, t_max=50, prev_n=None)
        bsf65.intersection(ray, t_min=0, t_max=50, prev_n=asf65.n_r)
            
        asf66.intersection(ray, t_min=0, t_max=50, prev_n=None)
        bsf66.intersection(ray, t_min=0, t_max=50, prev_n=asf66.n_r)
            
        asf67.intersection(ray, t_min=0, t_max=50, prev_n=None)
        bsf67.intersection(ray, t_min=0, t_max=50, prev_n=asf67.n_r)
            
        asf68.intersection(ray, t_min=0, t_max=50, prev_n=None)
        bsf68.intersection(ray, t_min=0, t_max=50, prev_n=asf68.n_r)
            
        asf69.intersection(ray, t_min=0, t_max=50, prev_n=None)
        bsf69.intersection(ray, t_min=0, t_max=50, prev_n=asf69.n_r)
        
        asf70.intersection(ray, t_min=0, t_max=50, prev_n=None)
        bsf70.intersection(ray, t_min=0, t_max=50, prev_n=asf70.n_r)

        asf71.intersection(ray, t_min=0, t_max=50, prev_n=None)
        bsf71.intersection(ray, t_min=0, t_max=50, prev_n=asf71.n_r)
            
        asf72.intersection(ray, t_min=0, t_max=50, prev_n=None)
        bsf72.intersection(ray, t_min=0, t_max=50, prev_n=asf72.n_r)
                    
        asf73.intersection(ray, t_min=0, t_max=50, prev_n=None)
        bsf73.intersection(ray, t_min=0, t_max=50, prev_n=asf73.n_r)
            
        asf74.intersection(ray, t_min=0, t_max=50, prev_n=None)
        bsf74.intersection(ray, t_min=0, t_max=50, prev_n=asf74.n_r)
           
        asf75.intersection(ray, t_min=0, t_max=50, prev_n=None)
        bsf75.intersection(ray, t_min=0, t_max=50, prev_n=asf75.n_r)
            
        asf76.intersection(ray, t_min=0, t_max=50, prev_n=None)
        bsf76.intersection(ray, t_min=0, t_max=50, prev_n=asf76.n_r)
            
        asf77.intersection(ray, t_min=0, t_max=50, prev_n=None)
        bsf77.intersection(ray, t_min=0, t_max=50, prev_n=asf77.n_r)
            
        asf78.intersection(ray, t_min=0, t_max=50, prev_n=None)
        bsf78.intersection(ray, t_min=0, t_max=50, prev_n=asf78.n_r)
            
        asf79.intersection(ray, t_min=0, t_max=50, prev_n=None)
        bsf79.intersection(ray, t_min=0, t_max=50, prev_n=asf79.n_r)
            
        asf80.intersection(ray, t_min=0, t_max=50, prev_n=None)
        bsf80.intersection(ray, t_min=0, t_max=50, prev_n=asf80.n_r)
            
        asf81.intersection(ray, t_min=0, t_max=50, prev_n=None)
        bsf81.intersection(ray, t_min=0, t_max=50, prev_n=asf81.n_r)
            
        asf82.intersection(ray, t_min=0, t_max=50, prev_n=None)
        bsf82.intersection(ray, t_min=0, t_max=50, prev_n=asf82.n_r)

        asf83.intersection(ray, t_min=0, t_max=50, prev_n=None)
        bsf83.intersection(ray, t_min=0, t_max=50, prev_n=asf83.n_r)
                    
        asf84.intersection(ray, t_min=0, t_max=50, prev_n=None)
        bsf84.intersection(ray, t_min=0, t_max=50, prev_n=asf84.n_r)
            
        asf85.intersection(ray, t_min=0, t_max=50, prev_n=None)
        bsf85.intersection(ray, t_min=0, t_max=50, prev_n=asf85.n_r)
            
        asf86.intersection(ray, t_min=0, t_max=50, prev_n=None)
        bsf86.intersection(ray, t_min=0, t_max=50, prev_n=asf86.n_r)
            
        asf87.intersection(ray, t_min=0, t_max=50, prev_n=None)
        bsf87.intersection(ray, t_min=0, t_max=50, prev_n=asf87.n_r)
            
        asf88.intersection(ray, t_min=0, t_max=50, prev_n=None)
        bsf88.intersection(ray, t_min=0, t_max=50, prev_n=asf88.n_r)
           
        asf89.intersection(ray, t_min=0, t_max=50, prev_n=None)
        bsf89.intersection(ray, t_min=0, t_max=50, prev_n=asf89.n_r)
            
        asf90.intersection(ray, t_min=0, t_max=50, prev_n=None)
        bsf90.intersection(ray, t_min=0, t_max=50, prev_n=asf90.n_r)
            
        asf91.intersection(ray, t_min=0, t_max=50, prev_n=None)
        bsf91.intersection(ray, t_min=0, t_max=50, prev_n=asf91.n_r)
            
        asf92.intersection(ray, t_min=0, t_max=50, prev_n=None)
        bsf92.intersection(ray, t_min=0, t_max=50, prev_n=asf92.n_r)
            
        asf93.intersection(ray, t_min=0, t_max=50, prev_n=None)
        bsf93.intersection(ray, t_min=0, t_max=50, prev_n=asf93.n_r)

        asf94.intersection(ray, t_min=0, t_max=50, prev_n=None)
        bsf94.intersection(ray, t_min=0, t_max=50, prev_n=asf94.n_r)
            
        asf95.intersection(ray, t_min=0, t_max=50, prev_n=None)
        bsf95.intersection(ray, t_min=0, t_max=50, prev_n=asf95.n_r)
            
        asf96.intersection(ray, t_min=0, t_max=50, prev_n=None)
        bsf96.intersection(ray, t_min=0, t_max=50, prev_n=asf96.n_r)
            
        asf97.intersection(ray, t_min=0, t_max=50, prev_n=None)
        bsf97.intersection(ray, t_min=0, t_max=50, prev_n=asf97.n_r)
            
        asf98.intersection(ray, t_min=0, t_max=50, prev_n=None)
        bsf98.intersection(ray, t_min=0, t_max=50, prev_n=asf98.n_r)
            
        asf99.intersection(ray, t_min=0, t_max=50, prev_n=None)
        bsf99.intersection(ray, t_min=0, t_max=50, prev_n=asf99.n_r)
            
        asf100.intersection(ray, t_min=0, t_max=50, prev_n=None)
        bsf100.intersection(ray, t_min=0, t_max=50, prev_n=asf100.n_r)
        '''
        asf101.intersection(ray, t_min=0, t_max=50, prev_n=None)
        bsf101.intersection(ray, t_min=0, t_max=50, prev_n=asf101.n_r)
        asf101.render(ax1)
        bsf101.render(ax1)
        
        asf102.intersection(ray, t_min=0, t_max=50, prev_n=None)
        bsf102.intersection(ray, t_min=0, t_max=50, prev_n=asf102.n_r)
        asf102.render(ax1)
        bsf102.render(ax1)
        
        asf103.intersection(ray, t_min=0, t_max=50, prev_n=None)
        bsf103.intersection(ray, t_min=0, t_max=50, prev_n=asf103.n_r)
        
        asf104.intersection(ray, t_min=0, t_max=50, prev_n=None)
        bsf104.intersection(ray, t_min=0, t_max=50, prev_n=asf104.n_r)

        asf105.intersection(ray, t_min=0, t_max=50, prev_n=None)
        bsf105.intersection(ray, t_min=0, t_max=50, prev_n=asf105.n_r)
        
        asf106.intersection(ray, t_min=0, t_max=50, prev_n=None)
        bsf106.intersection(ray, t_min=0, t_max=50, prev_n=asf106.n_r)

        asf107.intersection(ray, t_min=0, t_max=50, prev_n=None)
        bsf107.intersection(ray, t_min=0, t_max=50, prev_n=asf107.n_r)

        asf108.intersection(ray, t_min=0, t_max=50, prev_n=None)
        bsf108.intersection(ray, t_min=0, t_max=50, prev_n=asf108.n_r)

        asf109.intersection(ray, t_min=0, t_max=50, prev_n=None)
        bsf109.intersection(ray, t_min=0, t_max=50, prev_n=asf109.n_r)

        asf110.intersection(ray, t_min=0, t_max=50, prev_n=None)
        bsf110.intersection(ray, t_min=0, t_max=50, prev_n=asf110.n_r)
        
        asf111.intersection(ray, t_min=0, t_max=50, prev_n=None)
        bsf111.intersection(ray, t_min=0, t_max=50, prev_n=asf111.n_r)

        asf112.intersection(ray, t_min=0, t_max=50, prev_n=None)
        bsf112.intersection(ray, t_min=0, t_max=50, prev_n=asf112.n_r)

        asf113.intersection(ray, t_min=0, t_max=50, prev_n=None)
        bsf113.intersection(ray, t_min=0, t_max=50, prev_n=asf113.n_r)

        asf114.intersection(ray, t_min=0, t_max=50, prev_n=None)
        bsf114.intersection(ray, t_min=0, t_max=50, prev_n=asf114.n_r)

        asf115.intersection(ray, t_min=0, t_max=50, prev_n=None)
        bsf115.intersection(ray, t_min=0, t_max=50, prev_n=asf115.n_r)
        
        asf116.intersection(ray, t_min=0, t_max=50, prev_n=None)
        bsf116.intersection(ray, t_min=0, t_max=50, prev_n=asf116.n_r)

        asf117.intersection(ray, t_min=0, t_max=50, prev_n=None)
        bsf117.intersection(ray, t_min=0, t_max=50, prev_n=asf117.n_r)

        asf118.intersection(ray, t_min=0, t_max=50, prev_n=None)
        bsf118.intersection(ray, t_min=0, t_max=50, prev_n=asf118.n_r)

        asf119.intersection(ray, t_min=0, t_max=50, prev_n=None)
        bsf119.intersection(ray, t_min=0, t_max=50, prev_n=asf119.n_r)

        asf120.intersection(ray, t_min=0, t_max=50, prev_n=None)
        bsf120.intersection(ray, t_min=0, t_max=50, prev_n=asf120.n_r)
        
        asf121.intersection(ray, t_min=0, t_max=50, prev_n=None)
        bsf121.intersection(ray, t_min=0, t_max=50, prev_n=asf121.n_r)

        asf122.intersection(ray, t_min=0, t_max=50, prev_n=None)
        bsf122.intersection(ray, t_min=0, t_max=50, prev_n=asf122.n_r)

        asf123.intersection(ray, t_min=0, t_max=50, prev_n=None)
        bsf123.intersection(ray, t_min=0, t_max=50, prev_n=asf123.n_r)

        asf124.intersection(ray, t_min=0, t_max=50, prev_n=None)
        bsf124.intersection(ray, t_min=0, t_max=50, prev_n=asf124.n_r)

        asf125.intersection(ray, t_min=0, t_max=50, prev_n=None)
        bsf125.intersection(ray, t_min=0, t_max=50, prev_n=asf125.n_r)
    
        asf126.intersection(ray, t_min=0, t_max=50, prev_n=None)
        bsf126.intersection(ray, t_min=0, t_max=50, prev_n=asf126.n_r)
            
        asf127.intersection(ray, t_min=0, t_max=50, prev_n=None)
        bsf127.intersection(ray, t_min=0, t_max=50, prev_n=asf127.n_r)

        asf128.intersection(ray, t_min=0, t_max=50, prev_n=None)
        bsf128.intersection(ray, t_min=0, t_max=50, prev_n=asf128.n_r)

        asf129.intersection(ray, t_min=0, t_max=50, prev_n=None)
        bsf129.intersection(ray, t_min=0, t_max=50, prev_n=asf129.n_r)

        asf130.intersection(ray, t_min=0, t_max=50, prev_n=None)
        bsf130.intersection(ray, t_min=0, t_max=50, prev_n=asf130.n_r)

        asf131.intersection(ray, t_min=0, t_max=50, prev_n=None)
        bsf131.intersection(ray, t_min=0, t_max=50, prev_n=asf131.n_r)

        asf132.intersection(ray, t_min=0, t_max=50, prev_n=None)
        bsf132.intersection(ray, t_min=0, t_max=50, prev_n=asf132.n_r)

        asf133.intersection(ray, t_min=0, t_max=50, prev_n=None)
        bsf133.intersection(ray, t_min=0, t_max=50, prev_n=asf133.n_r)

        asf134.intersection(ray, t_min=0, t_max=50, prev_n=None)
        bsf134.intersection(ray, t_min=0, t_max=50, prev_n=asf134.n_r)

        asf135.intersection(ray, t_min=0, t_max=50, prev_n=None)
        bsf135.intersection(ray, t_min=0, t_max=50, prev_n=asf135.n_r)

        asf136.intersection(ray, t_min=0, t_max=50, prev_n=None)
        bsf136.intersection(ray, t_min=0, t_max=50, prev_n=asf136.n_r)

        asf137.intersection(ray, t_min=0, t_max=50, prev_n=None)
        bsf137.intersection(ray, t_min=0, t_max=50, prev_n=asf137.n_r)

        asf138.intersection(ray, t_min=0, t_max=50, prev_n=None)
        bsf138.intersection(ray, t_min=0, t_max=50, prev_n=asf138.n_r)

        asf139.intersection(ray, t_min=0, t_max=50, prev_n=None)
        bsf139.intersection(ray, t_min=0, t_max=50, prev_n=asf139.n_r)

        asf140.intersection(ray, t_min=0, t_max=50, prev_n=None)
        bsf140.intersection(ray, t_min=0, t_max=50, prev_n=asf140.n_r)

        asf141.intersection(ray, t_min=0, t_max=50, prev_n=None)
        bsf141.intersection(ray, t_min=0, t_max=50, prev_n=asf141.n_r)

        asf142.intersection(ray, t_min=0, t_max=50, prev_n=None)
        bsf142.intersection(ray, t_min=0, t_max=50, prev_n=asf142.n_r)

        asf143.intersection(ray, t_min=0, t_max=50, prev_n=None)
        bsf143.intersection(ray, t_min=0, t_max=50, prev_n=asf143.n_r)

        asf144.intersection(ray, t_min=0, t_max=50, prev_n=None)
        bsf144.intersection(ray, t_min=0, t_max=50, prev_n=asf144.n_r)

        asf145.intersection(ray, t_min=0, t_max=50, prev_n=None)
        bsf145.intersection(ray, t_min=0, t_max=50, prev_n=asf145.n_r)

        asf146.intersection(ray, t_min=0, t_max=50, prev_n=None)
        bsf146.intersection(ray, t_min=0, t_max=50, prev_n=asf146.n_r)

        asf147.intersection(ray, t_min=0, t_max=50, prev_n=None)
        bsf147.intersection(ray, t_min=0, t_max=50, prev_n=asf147.n_r)

        asf148.intersection(ray, t_min=0, t_max=50, prev_n=None)
        bsf148.intersection(ray, t_min=0, t_max=50, prev_n=asf148.n_r)

        asf149.intersection(ray, t_min=0, t_max=50, prev_n=None)
        bsf149.intersection(ray, t_min=0, t_max=50, prev_n=asf149.n_r)

        asf150.intersection(ray, t_min=0, t_max=50, prev_n=None)
        bsf150.intersection(ray, t_min=0, t_max=50, prev_n=asf150.n_r)

        asf151.intersection(ray, t_min=0, t_max=50, prev_n=None)
        bsf151.intersection(ray, t_min=0, t_max=50, prev_n=asf151.n_r)

        asf152.intersection(ray, t_min=0, t_max=50, prev_n=None)
        bsf152.intersection(ray, t_min=0, t_max=50, prev_n=asf152.n_r)

        asf153.intersection(ray, t_min=0, t_max=50, prev_n=None)
        bsf153.intersection(ray, t_min=0, t_max=50, prev_n=asf153.n_r)

        asf154.intersection(ray, t_min=0, t_max=50, prev_n=None)
        bsf154.intersection(ray, t_min=0, t_max=50, prev_n=asf154.n_r)

        asf155.intersection(ray, t_min=0, t_max=50, prev_n=None)
        bsf155.intersection(ray, t_min=0, t_max=50, prev_n=asf155.n_r)

        asf156.intersection(ray, t_min=0, t_max=50, prev_n=None)
        bsf156.intersection(ray, t_min=0, t_max=50, prev_n=asf156.n_r)

        asf157.intersection(ray, t_min=0, t_max=50, prev_n=None)
        bsf157.intersection(ray, t_min=0, t_max=50, prev_n=asf157.n_r)

        asf158.intersection(ray, t_min=0, t_max=50, prev_n=None)
        bsf158.intersection(ray, t_min=0, t_max=50, prev_n=asf158.n_r)

        asf159.intersection(ray, t_min=0, t_max=50, prev_n=None)
        bsf159.intersection(ray, t_min=0, t_max=50, prev_n=asf159.n_r)

        asf160.intersection(ray, t_min=0, t_max=50, prev_n=None)
        bsf160.intersection(ray, t_min=0, t_max=50, prev_n=asf160.n_r)

        asf161.intersection(ray, t_min=0, t_max=50, prev_n=None)
        bsf161.intersection(ray, t_min=0, t_max=50, prev_n=asf160.n_r)

        asf162.intersection(ray, t_min=0, t_max=50, prev_n=None)
        bsf162.intersection(ray, t_min=0, t_max=50, prev_n=asf162.n_r)

        asf163.intersection(ray, t_min=0, t_max=50, prev_n=None)
        bsf163.intersection(ray, t_min=0, t_max=50, prev_n=asf162.n_r)
        
        asf164.intersection(ray, t_min=0, t_max=50, prev_n=None)
        bsf164.intersection(ray, t_min=0, t_max=50, prev_n=asf163.n_r)
        
        asf165.intersection(ray, t_min=0, t_max=50, prev_n=None)
        bsf165.intersection(ray, t_min=0, t_max=50, prev_n=asf165.n_r)

        asf166.intersection(ray, t_min=0, t_max=50, prev_n=None)
        bsf166.intersection(ray, t_min=0, t_max=50, prev_n=asf166.n_r)

        asf167.intersection(ray, t_min=0, t_max=50, prev_n=None)
        bsf167.intersection(ray, t_min=0, t_max=50, prev_n=asf166.n_r)

        asf168.intersection(ray, t_min=0, t_max=50, prev_n=None)
        bsf168.intersection(ray, t_min=0, t_max=50, prev_n=asf168.n_r)

        asf169.intersection(ray, t_min=0, t_max=50, prev_n=None)
        bsf169.intersection(ray, t_min=0, t_max=50, prev_n=asf169.n_r)

        asf170.intersection(ray, t_min=0, t_max=50, prev_n=None)
        bsf170.intersection(ray, t_min=0, t_max=50, prev_n=asf170.n_r)

        asf171.intersection(ray, t_min=0, t_max=50, prev_n=None)
        bsf171.intersection(ray, t_min=0, t_max=50, prev_n=asf171.n_r)

        asf172.intersection(ray, t_min=0, t_max=50, prev_n=None)
        bsf172.intersection(ray, t_min=0, t_max=50, prev_n=asf172.n_r)

        asf173.intersection(ray, t_min=0, t_max=50, prev_n=None)
        bsf173.intersection(ray, t_min=0, t_max=50, prev_n=asf173.n_r)
        
        asf174.intersection(ray, t_min=0, t_max=50, prev_n=None)
        bsf174.intersection(ray, t_min=0, t_max=50, prev_n=asf174.n_r)
        asf174.render(ax1)
        bsf174.render(ax1)
                
        asf175.intersection(ray, t_min=0, t_max=50, prev_n=None)
        bsf175.intersection(ray, t_min=0, t_max=50, prev_n=asf175.n_r)
        asf175.render(ax1)
        bsf175.render(ax1)
                        
        asf176.intersection(ray, t_min=0, t_max=50, prev_n=None)
        bsf176.intersection(ray, t_min=0, t_max=50, prev_n=asf176.n_r)
        asf176.render(ax1)
        bsf176.render(ax1)
                        
        asf177.intersection(ray, t_min=0, t_max=50, prev_n=None)
        bsf177.intersection(ray, t_min=0, t_max=50, prev_n=asf177.n_r)
        asf177.render(ax1)
        bsf177.render(ax1)
                        
        asf178.intersection(ray, t_min=0, t_max=50, prev_n=None)
        bsf178.intersection(ray, t_min=0, t_max=50, prev_n=asf178.n_r)
        asf178.render(ax1)
        bsf178.render(ax1)
                        
        asf179.intersection(ray, t_min=0, t_max=50, prev_n=None)
        bsf179.intersection(ray, t_min=0, t_max=50, prev_n=asf179.n_r)
        asf179.render(ax1)
        bsf179.render(ax1)
                        
        asf180.intersection(ray, t_min=0, t_max=50, prev_n=None)
        bsf180.intersection(ray, t_min=0, t_max=50, prev_n=asf180.n_r)
        asf180.render(ax1)
        bsf180.render(ax1)
                        
        asf181.intersection(ray, t_min=0, t_max=50, prev_n=None)
        bsf181.intersection(ray, t_min=0, t_max=50, prev_n=asf181.n_r)
        asf181.render(ax1)
        bsf181.render(ax1)
                                
        asf182.intersection(ray, t_min=0, t_max=50, prev_n=None)
        bsf182.intersection(ray, t_min=0, t_max=50, prev_n=asf182.n_r)
        asf182.render(ax1)
        bsf182.render(ax1)
                                
        asf183.intersection(ray, t_min=0, t_max=50, prev_n=None)
        bsf183.intersection(ray, t_min=0, t_max=50, prev_n=asf183.n_r)
        asf183.render(ax1)
        bsf183.render(ax1)
                                
        asf184.intersection(ray, t_min=0, t_max=50, prev_n=None)
        bsf184.intersection(ray, t_min=0, t_max=50, prev_n=asf184.n_r)
        asf184.render(ax1)
        bsf184.render(ax1)
                                
        asf185.intersection(ray, t_min=0, t_max=50, prev_n=None)
        bsf185.intersection(ray, t_min=0, t_max=50, prev_n=asf185.n_r)
        asf185.render(ax1)
        bsf185.render(ax1)
                                
        asf186.intersection(ray, t_min=0, t_max=50, prev_n=None)
        bsf186.intersection(ray, t_min=0, t_max=50, prev_n=asf186.n_r)
        asf186.render(ax1)
        bsf186.render(ax1)
                                
        asf187.intersection(ray, t_min=0, t_max=50, prev_n=None)
        bsf187.intersection(ray, t_min=0, t_max=50, prev_n=asf187.n_r)
        asf187.render(ax1)
        bsf187.render(ax1)
                                
        asf188.intersection(ray, t_min=0, t_max=50, prev_n=None)
        bsf188.intersection(ray, t_min=0, t_max=50, prev_n=asf188.n_r)
        asf188.render(ax1)
        bsf188.render(ax1)
        '''             
        asf189.intersection(ray, t_min=0, t_max=50, prev_n=None)
        bsf189.intersection(ray, t_min=0, t_max=50, prev_n=asf189.n_r)
        asf189.render(ax1)
        bsf189.render(ax1)

        asf190.intersection(ray, t_min=0, t_max=50, prev_n=None)
        bsf190.intersection(ray, t_min=0, t_max=50, prev_n=asf190.n_r)
        asf190.render(ax1)
        bsf190.render(ax1)

        asf191.intersection(ray, t_min=0, t_max=50, prev_n=None)
        bsf191.intersection(ray, t_min=0, t_max=50, prev_n=asf191.n_r)
        asf191.render(ax1)
        bsf191.render(ax1)
        
        asf192.intersection(ray, t_min=0, t_max=50, prev_n=None)
        bsf192.intersection(ray, t_min=0, t_max=50, prev_n=asf192.n_r)
        asf192.render(ax1)
        bsf192.render(ax1)
                
        asf193.intersection(ray, t_min=0, t_max=50, prev_n=None)
        bsf193.intersection(ray, t_min=0, t_max=50, prev_n=asf193.n_r)
        asf193.render(ax1)
        bsf193.render(ax1)
                
        asf194.intersection(ray, t_min=0, t_max=50, prev_n=None)
        bsf194.intersection(ray, t_min=0, t_max=50, prev_n=asf194.n_r)
        asf194.render(ax1)
        bsf194.render(ax1)
                
        asf195.intersection(ray, t_min=0, t_max=50, prev_n=None)
        bsf195.intersection(ray, t_min=0, t_max=50, prev_n=asf195.n_r)
        asf195.render(ax1)
        bsf195.render(ax1)
                
        asf196.intersection(ray, t_min=0, t_max=50, prev_n=None)
        bsf196.intersection(ray, t_min=0, t_max=50, prev_n=asf196.n_r)
        asf196.render(ax1)
        bsf196.render(ax1)

        asf197.intersection(ray, t_min=0, t_max=50, prev_n=None)
        bsf197.intersection(ray, t_min=0, t_max=50, prev_n=asf197.n_r)
        asf197.render(ax1)
        bsf197.render(ax1)
                
        asf198.intersection(ray, t_min=0, t_max=50, prev_n=None)
        bsf198.intersection(ray, t_min=0, t_max=50, prev_n=asf199.n_r)
        asf198.render(ax1)
        bsf198.render(ax1)
                
        asf199.intersection(ray, t_min=0, t_max=50, prev_n=None)
        bsf199.intersection(ray, t_min=0, t_max=50, prev_n=asf199.n_r)
        asf199.render(ax1)
        bsf199.render(ax1)
                
        asf200.intersection(ray, t_min=0, t_max=50, prev_n=None)
        bsf200.intersection(ray, t_min=0, t_max=50, prev_n=asf200.n_r)
        asf200.render(ax1)
        bsf200.render(ax1)
                        
        asf201.intersection(ray, t_min=0, t_max=50, prev_n=None)
        bsf201.intersection(ray, t_min=0, t_max=50, prev_n=asf201.n_r)
        asf201.render(ax1)
        bsf201.render(ax1)
                        
        asf202.intersection(ray, t_min=0, t_max=50, prev_n=None)
        bsf202.intersection(ray, t_min=0, t_max=50, prev_n=asf202.n_r)
        asf202.render(ax1)
        bsf202.render(ax1)
                        
        asf203.intersection(ray, t_min=0, t_max=50, prev_n=None)
        bsf203.intersection(ray, t_min=0, t_max=50, prev_n=asf203.n_r)
        asf203.render(ax1)
        bsf203.render(ax1)
                        
        asf204.intersection(ray, t_min=0, t_max=50, prev_n=None)
        bsf204.intersection(ray, t_min=0, t_max=50, prev_n=asf204.n_r)
        asf204.render(ax1)
        bsf204.render(ax1)
                        
        asf205.intersection(ray, t_min=0, t_max=50, prev_n=None)
        bsf205.intersection(ray, t_min=0, t_max=50, prev_n=asf205.n_r)
        asf205.render(ax1)
        bsf205.render(ax1)
                                
        asf206.intersection(ray, t_min=0, t_max=50, prev_n=None)
        bsf206.intersection(ray, t_min=0, t_max=50, prev_n=asf206.n_r)
        asf206.render(ax1)
        bsf206.render(ax1)
                                
        asf207.intersection(ray, t_min=0, t_max=50, prev_n=None)
        bsf207.intersection(ray, t_min=0, t_max=50, prev_n=asf207.n_r)
        asf207.render(ax1)
        bsf207.render(ax1)
                                
        asf208.intersection(ray, t_min=0, t_max=50, prev_n=None)
        bsf208.intersection(ray, t_min=0, t_max=50, prev_n=asf208.n_r)
        asf208.render(ax1)
        bsf208.render(ax1)
        '''

        ray.render_all(ax1, time_of_flights = 5)

#ax.set_title("Focus")
ax1.set_xlabel('x, m')
ax1.set_ylabel('y, um')

cursor = Cursor(ax1, horizOn = True, vertOn = True, linewidth=1.0, color = '#FF6500')

def onPress(event):
    print('Coordinate Position {0}, {1}'.format(event.x, event.y))
    print('Data Point Value x: {0} m, y: {1} m'.format(round(event.xdata, 9), event.ydata))
    plt.plot(round(event.xdata, 9), event.ydata, ',')# 'o', markersize = 10, color = 'red', label = f'x:{event.xdata} \ny:{event.ydata}')

fig.canvas.mpl_connect('button_press_event', onPress)
 
ax2 = plt.subplot2grid((3,4), (2,0), colspan = 2)
#ax2 = plt.subplots(2,1)
ax2.set_axis_off()
ax2 = plt.text(0, 0, f"n = {round(material_n, 5)} \nwavelength = {wavelength_a}, A \ndelta = {delta} \nmu = {total_linear_coeff} \nКол-во линз в трансфокаторе 1: {N1} \nКол-во линз в трансфокаторе 2: {N2} \nРасстояние между линзами: {dist_crl} мкм", ha='left')

#from matplotlib.widgets import MultiCursor
#cursor = MultiCursor(fig.canvas, ax1, color='r',lw=0.5, horizOn=True, vertOn=True)

plt.show()

#print('Focal point: ', 'min: ', round(min(f_point), 9), "m", 'max: ', round(max(f_point), 9), "m", 'mean:', round((max(f_point)+min(f_point))/2, 9), "m")
#print('Aeff, um', 'min:', min(Aeff)*1e6, 'max:', max(Aeff)*1e6, 'mean:', (max(Aeff)+min(Aeff))/2*1e6)
#print('N.A., rad', 'min:', min(numerical_aperture), 'mean:', (max(numerical_aperture)+min(numerical_aperture))/2, 'max:', max(numerical_aperture))
#print('Depth of Field, um', 'min:', min(depth_of_field)*1e6, 'max:',max(depth_of_field)*1e6, 'mean:', (max(depth_of_field) + min(depth_of_field))/2*1e6)
#print("Spherical aberration size: ", dof*1e6, "um")
#print("Res: ", Res, "m")

