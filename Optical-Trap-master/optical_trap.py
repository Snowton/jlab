import os
import glob
import re
import warnings
import numpy as np
import random
import sys
# from skimage.external import tifffile
from skimage.io import imsave

def match_filename(pattern, root_dir):
    '''returns list of matched patterns in file names

    inputs
    ------
    pattern - raw text indicating the regex pattern to use for the file names
          The part in the string to extract should be denoted with parentheses ()
    root_dir - directory to search the files

    outputs
    matches - list of strings that are file names that match the given pattern             
    -------

    '''
    file_names = glob.glob(os.path.join(root_dir, "*"))
    matches = [re.fullmatch(pattern, os.path.basename(file_name)) for file_name in file_names]
    matches = [match.group(1) for match in matches if match is not None]

    return matches

k_B = 1.38064852*10**-23

#####################################################################
T, laser_power, particle_radius, sampling_rate, frame_rate, total_time  = [float(s) for s in input('Input: temperature(K), laser_power(mA), particle_radius(um), sampling_rate(Hz), frame_rate(Hz), total_time(s)\n').split()]

if(273.15>T):
    print('temperature is too low')
    sys.exit()
if(373.15<T):
    print('temperature is too high')
    sys.exit()    
if(50>laser_power):
    print('laser_power is too low')
    sys.exit()
if(500<laser_power):
    print('laser_power is too high')
    sys.exit()
if(0.5>particle_radius):
    print('particle_radius is too small')
    sys.exit()
if(2.5<particle_radius):
    print('particle_radius is too large')
    sys.exit()
if(50000<sampling_rate):
    print('sampling_rate is too high')
    sys.exit()
if(50<frame_rate):
    print('frame_rate is too high')
    sys.exit()
if(1/np.minimum(sampling_rate,frame_rate) > total_time):
    print('recording period is longer than the total_time')
    sys.exit()


root_dir = './data/'
if(os.path.isdir(root_dir)==False):
    os.mkdir(root_dir)

# set meta file name
nums = match_filename(r'exp_stack_([0-9]+).tif', root_dir=root_dir)
nums = [int(num) for num in nums]; idx0 = max(nums) + 1 if nums else 0
file_name_stack = 'exp_stack_{}.tif'.format(idx0)
file_name_list = 'exp_list_{}.csv'.format(idx0)
file_path_stack = os.path.join(root_dir, file_name_stack)
file_path_list = os.path.join(root_dir, file_name_list)

# set simulation rate with a minimum 1000Hz
if(sampling_rate < 5000):
    simulation_rate = 5000
else:
    simulation_rate = sampling_rate

# define simulation paramteres
dt = 1/simulation_rate
data_len = int(total_time*simulation_rate)
gamma = 6*np.pi*10**-3*particle_radius
k = (laser_power/1000)/gamma
diff_coef = T/gamma

# simulate the trapped particle using first order stochastic Euler method
force_matrix = np.array([[-k,0],[0,-k]])
diffusion_matrix = np.array([[diff_coef,0],[0,diff_coef]])
prefactor1 = force_matrix * dt
prefactor2 = np.sqrt(2 * diffusion_matrix * dt)
n_steps_initial = 10000
x0 = np.zeros((2, 1)).astype('float32')
processes = np.empty((2, data_len)).astype('float32')
for idx in range(n_steps_initial):
    x0 = x0 + np.matmul(prefactor1,x0) + np.matmul(prefactor2,np.random.normal(0,1,[2,1]))
    
processes[:, 0] = x0.T
x = x0
for idx in range(data_len-1):
    x = x + np.matmul(prefactor1,x) + np.matmul(prefactor2,np.random.normal(0,1,[2,1]))
    processes[:, idx+1] = x.T
processes = processes * np.sqrt(k_B) * 10**3 * 10**6

# save the QPD signal accroding to the sampling frequency
np.savetxt(file_path_list, processes[:,0:-1:int(simulation_rate/sampling_rate)], delimiter=',')

# convert the trajectory signal into image signal
x_range = np.arange(-6.35, 6.4, 0.1) 
y_range = np.arange(-6.35, 6.4, 0.1)
xx, yy = np.meshgrid(x_range, y_range, sparse=True)  
sigma = particle_radius
imgs = []
for idx1 in range(0,data_len-1,int(simulation_rate/frame_rate)):
    '''
    simulation a fluorescent particle with imaging noise using a Gaussian model
    '''
    z = np.exp(-( (xx-processes[0,idx1])**2 + (yy-processes[1,idx1])**2)/(2*sigma)) * 200 + 20* np.random.normal(0,1,[128,128])
    np.clip(z, 0, 255, out=z)
    z = z.astype('uint8')
    imgs.append(z)
    
img_stack = np.stack(imgs, axis=0)

imsave(file_path_stack, img_stack)




