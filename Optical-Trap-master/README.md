# Optical-Trap

This is the Python code for virtualized optical trap experiment, the code in written in Python3 and to run the code you will need NumPy and scikit-image package. And If you installed the Anaconda distribution of Python, both NumPy and scikit-image come pre-installed and no further installation steps are necessary

After pull the code into your loacl directory, you can simply run it on your terminal: python optical_trap.
And it will ask you for six inputs as following: temperature(K), laser_power(mA), particle_radius(um), sampling_rate(Hz), frame_rate(Hz), total_time(s).

The inputs are limited with a certain range for pyhsical reasons as well as to mimic the real experimental set-up:

273.15(K)<temperature<373.15(K)

50(mA)<laser power<500(mA)

0.5(um)<particle radius<2.5(um)

sampling rate<50000(Hz)

frame rate<50(Hz)

An example of inputs are: temperature 300(K), laser power 100(mA), particle radius 1(um), sampling rate 5000(Hz), frame rate 50(Hz) and total time 60(s). Note that the trap stiffness is proportional to the laser power with a factor 10^-9(N/m*mA).

When the code finish running, a folder named data will be automatically generated containing a csv file mimicing the high frequency quadrant photodetector (QPD) positional data(callibrated and in units of um) and a tif file simulating the low frequency video microscopic data(with the pixel size to be 100nm). 

The QPD positional data can be taken directly to measure Boltzmann constant using the Equipartition theorem or the power spectrum density(PSD). More detailed information about these analysis are incorporated inside the lab mannual and here are a few more references about the PSD analysis:
https://www.osapublishing.org/DirectPDFAccess/C0023BE9-D621-D807-33B887CD12C872E8_119802/oe-14-25-12517.pdf?da=1&id=119802&seq=0&mobile=no
https://www.osapublishing.org/DirectPDFAccess/BFC68408-A69A-88D4-2C0BAB26797432C9_196904/oe-18-8-7670.pdf?da=1&id=196904&seq=0&mobile=no

After a successful measurement of the Boltzmann constant from the QPD positional data, you can do the same task on the video microscopic data. Normally, video microscopic signal has a much lower freqnency and a lot more noise than the QPD signal, so the biggest challenge here will be localize the center of the trapped particle. As I mimic it as a fluorescent video, the intensity profile will be a 2D-Gaussian centered at the particle position with background added. 

To do the localization task, there are many different methods. For example, as the intensity profile is a Gaussian, the maximun intensity pixel can be approximated at the center. However, as noise exists, the maximum intensity method may not be rouboust and a Gaussian fit might work better. 
You are also welcomed to use existing functions to help with the localization like Countour finding, Filtering regional maxima, Circular and Elliptical Hough Transforms etc. inside the scikit-image package for python(https://scikit-image.org). 
And the ImageJ python wrapper(https://github.com/imagej/pyimagej), the bwmorph function for Matlab, ContourDetect for Mathematica etc. 
Lastly, there are also open-source packages specialized in particle localization and tracking, for example https://www.nature.com/articles/nmeth.2071.
All methods are welcomed and also your own imagination!

In the end, you can compare you results from both sets of data.
Below is a link to my slide presentation of the virtual optical trap experiment with some of my own analysis:
https://www.dropbox.com/s/cij0hxty08vfw0v/20200916_junior_lab_optical_trap.pdf?dl=0

If you still feel ambitious after all the analysis, great! We have the next virtual experiment for you to design a microscopic heat engine out of the optical trap system. And more details are in the second half of the slides.

-Junang
