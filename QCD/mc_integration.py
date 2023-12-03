from numpy import random
import numpy as np
from scipy.special import gamma
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt 
import time


class MC:
	'''
	Calculate the volume (area - in 2 dimension) of an n-dimensional sphere

	'''
	def __init__(self, dimension, number_of_integration, throws):

		self.d = dimension
		self.n = number_of_integration
		self.t = throws
		self.err = []

	def area(self, radius=1):
		'''
		Generate random sampling and perform Monte Carlo integration. 
		Calculate the relative error using the formula for the volume of the n-dimensional sphere.
		'''
		inside_circle = 0
		i = 0
		while i < self.t:
			coordinates = np.array([random.uniform(-radius,radius) for x in range(self.d)])
			if np.square(coordinates).sum() <= radius**self.d:
				inside_circle += 1
			i += 1

		calculated_area = (((2 * radius) ** self.d) * inside_circle)/self.t
		relative_error = np.abs((calculated_area-np.pi**(self.d/2)/gamma(self.d/2+1))/np.pi**(self.d/2)/gamma(self.d/2+1))
		return [calculated_area, relative_error]

	
	def integration(self, func=area):
		'''
		Perform the Monte Carlo integration n times and averaging the relative error.
		'''
		relative_error_list = []
		for iteration in range(self.n):
			relative_error_list.append(func(self)[1])

		relative_error_avg = np.array(relative_error_list).sum()/len(relative_error_list)
		return relative_error_avg


def sample_generator(start, end):
	'''
	Generate the number of points for the Monte Carlo integration.
	'''
	sample = []
	for i in range(start,end,1):
		for k in range(10**i,10**(i+1),10**i):
			sample.append(k)
	return sample

def f(x, a):
	'''
	Define the function for scaling the relative error.
	'''
	return a * 1/np.sqrt(x) 



start = time.time()	

# Calculating the volume in 2,3,4,5,6 dimension using 1e5 sampling points
#dim2 = MC(2,1,1e5).area()
dim3 = MC(3,1,1e5).area()
dim4 = MC(4,1,1e5).area()
dim5 = MC(5,1,1e5).area()
dim6 = MC(6,1,1e5).area()
print("The volume of the unit circle in\n3D: ", dim3[0], "\n4D: ", dim4[0], "\n5D: ", dim5[0], "\n6D: ", dim6[0])
	
# generate a list of sampling points		
N = np.logspace(0, 3, num=1000)

# calculate relative error in 3 dimension
error_list3 = []
error_list_avg3 = []
count = 0
for i in N:
	error_list3.append(MC(3,1,i).integration()) # no averaging
	error_list_avg3.append(MC(3,1000,i).integration()) # averaging 1000 integration
	count +=1
	if count % (len(N)/10) == 0:
		print("iteration number ", i)

popt3, pcov3 = curve_fit(f, N, error_list3, p0=[0.55])
poptavg3, pcovavg3 = curve_fit(f, N, error_list_avg3, p0=[0.67])


plt.plot(N, f(N, *popt3), 'r-',label=r'$f(x)=a \cdot \frac{1}{\sqrt{N}}$,   where a=%5.3f' % tuple(popt3))
plt.plot(N,error_list3)
plt.xlabel("N (Number of sampling points) [1]")
plt.ylabel(r'$\sigma=\left | \frac{numerical-analytical}{analytical} \right |$ (Relative error) [1]')
plt.title("3D")
plt.xscale("log")
plt.yscale("log")
plt.xlim(1, )
plt.grid()
plt.legend()
plt.savefig('error3.png')
plt.close()

plt.plot(N, f(N, *poptavg3), 'r-',label=r'$f(x)=a \cdot \frac{1}{\sqrt{N}}$,   where a=%5.3f' % tuple(poptavg3))
plt.plot(N,error_list_avg3)
plt.xlabel("N (Number of sampling points) [1]")
plt.ylabel(r'$\sigma=\left | \frac{numerical-analytical}{analytical} \right |$ (Relative error) [1]')
plt.title("3D")
plt.xscale("log")
plt.yscale("log")
plt.xlim(1, )
plt.grid()
plt.legend()
plt.savefig('error_avg3.png')
plt.close()




# calculate relative error in 4 dimension
error_list4 = []
error_list_avg4 = []
count = 0
for i in N:
	error_list4.append(MC(4,1,i).integration()) # no averaging
	error_list_avg4.append(MC(4,1000,i).integration()) # averaging 1000 integration
	count +=1
	if count % (len(N)/10) == 0:
		print("iteration number ", i)

popt4, pcov4 = curve_fit(f, N, error_list4, p0=[0.55])
poptavg4, pcovavg4 = curve_fit(f, N, error_list_avg4, p0=[0.67])



plt.plot(N, f(N, *popt4), 'r-',label=r'$f(x)=a \cdot \frac{1}{\sqrt{N}}$,   where a=%5.3f' % tuple(popt4))
plt.plot(N,error_list4)
plt.xlabel("N (Number of sampling points) [1]")
plt.ylabel(r'$\sigma=\left | \frac{numerical-analytical}{analytical} \right |$ (Relative error) [1]')
plt.title("4D")
plt.xscale("log")
plt.yscale("log")
plt.xlim(1, )
plt.grid()
plt.legend()
plt.savefig('error4.png')
plt.close()

plt.plot(N, f(N, *poptavg4), 'r-',label=r'$f(x)=a \cdot \frac{1}{\sqrt{N}}$,   where a=%5.3f' % tuple(poptavg4))
plt.plot(N,error_list_avg4)
plt.xlabel("N (Number of sampling points) [1]")
plt.ylabel(r'$\sigma=\left | \frac{numerical-analytical}{analytical} \right |$ (Relative error) [1]')
plt.title("4D")
plt.xscale("log")
plt.yscale("log")
plt.xlim(1, )
plt.grid()
plt.legend()
plt.savefig('error_avg4.png')
plt.close()
end = time.time()-start
print('runtime:' + str(end))



