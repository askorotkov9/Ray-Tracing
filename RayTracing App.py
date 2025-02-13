import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import Cursor

from lens_params import *
from source import *
from ray import *
from surface import *


#Determines the quality of the plot
point_num=120

#output index of refraction and total linear coefficient from csv
dataframe = pd.read_csv('C:/Users/askor/OneDrive/Рабочий стол/Xoptics/2024/RayTracing App/materials/Be 2keV-100keV.csv', sep = ';')

index_n = dataframe[dataframe.isin([E]).any(axis=1)]
delta = 3.976746331307E-6  #float(index_n['Delta']) #searching for the delta by the input energy
material_n = 1 - delta	#float(index_n['n'])
wavelength_a = float(index_n['Wavelength, A']) #searching for the wavelength by the input energy 
total_linear_coeff = float(index_n['Linear attenuation coefficient (photoelectric part), 1/um']) #searching for the total linear coefficient by the input energy


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