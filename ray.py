import numpy as np
import math
from matplotlib.axes import Axes

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