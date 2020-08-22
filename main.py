# main.py

import numpy as np
from scipy.optimize import newton
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def F(E,e,M):
	return E-e*np.sin(E)-M

def Fp(E,e,M):
	 return 1-e*np.cos(E)

# Find the true anomaly at t hours after perigee passage

# setup
Re = 6378e3		# [m] radius of Earth
a = 4*Re		# [m] semimajor axis
rp = 1.5*Re		# [m] perigee radius
mu = 3.986e5	# [km^3*s^-2] G*m constant for Earth
t = np.linspace(0,12,500)*3600.0 # [s] time vector 
#t = t[-1] # debug
n = len(t)

# compute e
e = 1 - rp/a

# pre-allocate results vectors
f = np.zeros([n])
r = np.zeros([n])

flag = False
for i in np.arange(n):

	# compute Ms
	M = (mu/((a/1000.0)**3.0))**(0.5)*t[i]

	# compute Es (E0 = M)
	E = newton(F,M,fprime=Fp,args=(e,M),tol=1.0e-10,full_output=False)

	# compute f (radians)
	f[i] = 2*np.arctan((((1+e)/(1-e))**0.5)*np.tan(E/2))

	# compute r (radius)
	r[i] = a*Fp(E,e,M)
	#print(f[i],f[i]*180.0/np.pi,r[i]/1000.0)

	#print(f[i])

	# stop after one complete orbit
	if f[i] < 0.0:
		flag=True
	if f[i] > 0.0 and flag:
		# remove final iteration
		f = f[:i-1]
		r = r[:i-1]
		n = i-1
		print('break!')
		break


fig = plt.figure(num=None, figsize=(10, 10), dpi=150, facecolor='w', edgecolor='k')
ax = fig.add_axes([0.1, 0.1, 0.8, 0.8], polar=True)
xdata, ydata = [], []
ln, = plt.polar([],[],linestyle='-',color='gray')
ln2, = plt.polar([],[],'.k',ms=12)
an = plt.annotate('time = %5.1f hours\nradius = %5.1fe3 km'%((t[0]/3600),r[0]/1e6),(0,0),xytext=(np.pi*7/4,2.5))

def init():
    #ax.set_xlim(0, 2*np.pi)
    #ax.set_ylim(0,2)
    ax.clear()
    xdata, ydata = [], []
    plt.polar(np.linspace(0,2*np.pi,361),np.ones([361]),'-b')
    ax.set_rmax(7.0)
    ax.set_rticks(np.arange(7))
    #ax.set_title('Elliptical orbit about the Earth\nsemimajor axis = %.3f\nperigee radius = %.3f\n'%(a/Re,rp/Re),fontsize=10)
    ax.set_title('semimajor axis = %.3f, perigee radius = %.3f'%(a/Re,rp/Re),fontsize=10)

    return ln,

def update(frame):
    xdata.append(f[frame])
    ydata.append(r[frame]/Re)
    ln.set_data(xdata, ydata)
    ln2.set_data(f[frame],r[frame]/Re)
    an.set_text('time = %5.1f hours\nradius = %5.1fe3 km'%((t[frame]/3600),r[frame]/1e6))
    return ln,ln2,an

ani = FuncAnimation(fig, update, frames=np.arange(n),
                    init_func=init, blit=True,
                    interval=25)#,save_count=50)

plt.show()