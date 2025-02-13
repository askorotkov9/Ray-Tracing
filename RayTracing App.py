import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from typing import Union, Callable
from scipy.optimize import fsolve
from matplotlib.widgets import Cursor
import math

from lens_params import *
from source import *


#Determines the quality of the plot
point_num=120

#output index of refraction and total linear coefficient from csv
dataframe = pd.read_csv('C:/Users/askor/OneDrive/Рабочий стол/Xoptics/2024/RayTracing App/materials/Be 2keV-100keV.csv', sep = ';')

index_n = dataframe[dataframe.isin([E]).any(axis=1)]
delta = 3.976746331307E-6  #float(index_n['Delta']) #searching for the delta by the input energy
material_n = 1 - delta	#float(index_n['n'])
wavelength_a = float(index_n['Wavelength, A']) #searching for the wavelength by the input energy 
total_linear_coeff = float(index_n['Linear attenuation coefficient (photoelectric part), 1/um']) #searching for the total linear coefficient by the input energy

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


#Define a divergent bunch of rays
ray_number = 5
ray_number_divergence = 5
#divergence = np.linspace(-1E-6, 1E-6, num = ray_number)
y_start = np.linspace(-source_size/2, source_size/2, num = ray_number)
x_start = np.zeros_like(y_start)
theta = np.linspace(-divergence/2, divergence/2, num = ray_number_divergence)
rays = []




#TF's lenses set initialization

def asf(aperture, nr, x_c, R, y0):
	return Surface(lens_aperture = aperture, material_nr = nr, x_c = x_c, y0 = y0, R = R)
def bsf(aperture, nr, x_c, R, y0):
	return Surface(lens_aperture = aperture, material_nr = nr, x_c = x_c, y0 = y0, R = R)


tf1_lens_number = 60
tf2_lens_number = 100
tf1_start = 27
tf2_start = 0.2604
dist_crl = 600*1e-6


#TF1
asf1_1 = asf(aperture = R500['aperture']/2, nr = material_n, x_c = tf1_start, y0=y0, R = -R500['radius']) #воздух - линза
bsf1_1 = bsf(aperture = R500['aperture']/2, nr = 1.0, x_c = asf1_1.get_surface_apex(x_c) + d/2, y0=y0, R = R500['radius']) #воздух - линза (Be)

tf1_set = {'asf1_1': asf1_1, 'bsf1_1': bsf1_1}


for i in range(2, tf1_lens_number + 1):
    asf_xc = 2 * tf1_set[f'bsf1_{i-1}'].get_surface_apex(R500['aperture']) - tf1_set[f'asf1_{i-1}'].get_surface_apex(x_c) + dist_crl
    tf1_set[f'asf1_{i}'] = asf(aperture = R500['aperture']/2, nr = material_n, x_c = asf_xc, y0=y0, R = -R500['radius']) #воздух - линза
    bsf_xc = tf1_set[f'asf1_{i}'].get_surface_apex(x_c) + d/2
    tf1_set[f'bsf1_{i}'] = bsf(aperture = R500['aperture']/2, nr = 1.0, x_c = bsf_xc, y0=y0, R = R500['radius']) #воздух - линза (Be)


#TF2
asf2_1 = asf(aperture = R50['aperture']/2, nr = material_n, x_c = tf2_start, y0=y0, R = -R50['radius']) #воздух - линза
bsf2_1 = bsf(aperture = R50['aperture']/2, nr = 1.0, x_c = asf2_1.get_surface_apex(x_c) + d/2, y0=y0, R = R50['radius']) #воздух - линза (Be)

tf2_set = {'asf2_1': asf2_1, 'bsf2_1': bsf2_1}


for i in range(2, tf2_lens_number + 1):
    asf_xc = 2 * tf2_set[f'bsf2_{i-1}'].get_surface_apex(R500['aperture']) - tf2_set[f'asf2_{i-1}'].get_surface_apex(x_c) + dist_crl
    tf2_set[f'asf2_{i}'] = asf(aperture = R50['aperture']/2, nr = material_n, x_c = asf_xc, y0=y0, R = -R50['radius']) #воздух - линза
    bsf_xc = tf2_set[f'asf2_{i}'].get_surface_apex(x_c) + d/2
    tf2_set[f'bsf2_{i}'] = bsf(aperture = R50['aperture']/2, nr = 1.0, x_c = bsf_xc, y0=y0, R = R50['radius']) #воздух - линза (Be)



fig = plt.figure(figsize=(8, 5))
ax1 = plt.subplot2grid((3,4), (0, 0), colspan = 4, rowspan = 2)
ax1.grid()



tf1_input = []
tf2_input = [2, 5, 6, 7, 9, 11, 14, 16, 17, 18, 22, 25, 26, 30, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 43, 45, 46, 47, 49, 50]

tf1_set_input = {}
tf2_set_input = {}



for i in tf1_input:
	tf1_set_input[f'asf1_{i}'] = tf1_set[f'asf1_{i}']
	tf1_set_input[f'bsf1_{i}'] = tf1_set[f'bsf1_{i}']

for i in tf2_input:
	tf2_set_input[f'asf2_{i}'] = tf2_set[f'asf2_{i}']
	tf2_set_input[f'bsf2_{i}'] = tf2_set[f'bsf2_{i}']

	

def ray_tf1_intersection(ray, g):
	tf1_set[f'asf1_{g}'].intersection(ray, t_min=0, t_max=50, prev_n=None)
	tf1_set[f'bsf1_{g}'].intersection(ray, t_min=0, t_max=50, prev_n=tf1_set[f'asf1_{g}'].n_r)
	tf1_set[f'asf1_{g}'].render(ax1)
	tf1_set[f'bsf1_{g}'].render(ax1)



def ray_tf2_intersection(ray, l):
	tf2_set[f'asf2_{l}'].intersection(ray, t_min=0, t_max=50, prev_n=None)
	tf2_set[f'bsf2_{l}'].intersection(ray, t_min=0, t_max=50, prev_n=tf2_set[f'asf2_{l}'].n_r)
	tf2_set[f'asf2_{l}'].render(ax1)
	tf2_set[f'bsf2_{l}'].render(ax1)



for i in range(y_start.shape[0]):
    for j in range(theta.shape[0]):
        ray_wavelength = 0.1
        ray = Ray(x_start[i], y_start[i], theta[j], ray_wavelength)
        rays.append(ray)

        if len(tf1_set_input) != 0:
            length_tf1 = round(len(tf1_set_input)/2, 0)
            for g in range(1, int(length_tf1) + 1):
                ray_tf1_intersection(ray, g)
        if len(tf2_set_input) != 0:
            length_tf2 = round(len(tf2_set_input)/2, 0)
            for l in range(1, int(length_tf2) + 1):
                ray_tf2_intersection(ray, l)
        
        ray.render_all(ax1, time_of_flights = 2)


ax1.set_xlabel('x, m')
ax1.set_ylabel('y, um')

cursor = Cursor(ax1, horizOn = True, vertOn = True, linewidth=1.0, color = '#FF6500')

def onPress(event):
    print('Coordinate Position {0}, {1}'.format(event.x, event.y))
    print('Data Point Value x: {0} m, y: {1} m'.format(round(event.xdata, 9), event.ydata))
    plt.plot(round(event.xdata, 9), event.ydata, ',')

fig.canvas.mpl_connect('button_press_event', onPress)
 
ax2 = plt.subplot2grid((3,4), (2,0), colspan = 2)
ax2.set_axis_off()
ax2 = plt.text(0, 0, f"n = {round(material_n, 5)} \nwavelength = {wavelength_a}, A \ndelta = {delta} \nКол-во линз в TF1: {len(tf1_input)} \nКол-во линз в TF2: {len(tf2_input)} \nРасстояние между линзами: {dist_crl} м", ha='left')

plt.show()