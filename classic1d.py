import matplotlib
matplotlib.use('macosx')

import matplotlib.pyplot as plt

import numpy as np
import matplotlib.animation as animation


def iso1d(x1f,rho,rhou1,cs,cfl):
  nx1=len(x1f)-1
  dx1=x1f[1:nx1+1]-x1f[0:nx1]
  x1c=(x1f[0:nx1]+x1f[1:nx1+1])/2.
  #face velocity
  ux1f = np.zeros(nx1+1)
  #get dt with cfl
  print(len(dx1),len(rhou1),len(rho))
  dum1 = dx1/(cs+abs(rhou1/rho))
  dt1 = cfl * min(dum1)
  dt = dt1
  for ix1 in range(1,nx1):
    ux1f[ix1] = 0.5 * (rhou1[ix1]/rho[ix1]\
             +rhou1[ix1-1]/rho[ix1-1])
  #boundary u = 0
  ux1f[0]=0.
  ux1f[nx1]=0.
  #rho flux donor cell
  fluxrho1 = np.zeros(nx1+1)
  for ix1 in range(1,nx1):
    if ux1f[ix1] > 0.:
      fluxrho1[ix1] = rho[ix1-1] * ux1f[ix1]
    else:
      fluxrho1[ix1] = rho[ix1] * ux1f[ix1]
  #rho flux boundary 0
  #update rho
  for ix1 in range(0,nx1):
    rho[ix1] = rho[ix1] - dt * (fluxrho1[ix1+1]-\
               fluxrho1[ix1]) / dx1[ix1]
  #rho u flux donor cell
  fluxrhou1 = np.zeros(nx1+1)
  for ix1 in range(1,nx1):
    if ux1f[ix1] > 0.:
      fluxrhou1[ix1] = rhou1[ix1-1] * ux1f[ix1]
    else:
      fluxrhou1[ix1] = rhou1[ix1] * ux1f[ix1]
  #update rho u
  for ix1 in range(0,nx1):
    rhou1[ix1] = rhou1[ix1] - dt * (fluxrhou1[ix1+1]-\
               fluxrhou1[ix1]) / dx1[ix1]

  #source term from pressure
  p = rho*cs*cs
  for ix1 in range(1,nx1-1):
    rhou1[ix1] = rhou1[ix1] - dt * (p[ix1+1] - p[ix1-1])\
                /(x1c[ix1+1]-x1c[ix1-1])
  #boundary condition for pressure
  rhou1[0] = rhou1[0] - 0.5*dt*(p[1]-p[0])\
             /(x1c[1]-x1c[0])
  rhou1[nx1-1] = rhou1[nx1-1] - 0.5*dt*(p[nx1-1]-p[nx1-2])\
               /(x1c[nx1-1]-x1c[nx1-2])

  return dt 


#initial
N=50

x1f=np.linspace(1,101,N+1)

x1c=(x1f[0:N]+x1f[1:N+1])/2.

print(x1c)

rho=np.zeros(N)
rhou=np.zeros(N)
xmid=0.5*(max(x1f)*min(x1f))
dg=0.1*(max(x1f)-min(x1f))
rho=1.+0.3*np.exp(-(x1c-xmid)**2/dg**2)
cs=0.1


count=0
while count<3:

  cfl=0.4
  dt=iso1d(x1f,rho,rhou,cs,cfl)
  plt.plot(x1c,rhou/rho)
  count+=1
  plt.show() 


# set up figure and animation
fig = plt.figure()
ax = fig.add_subplot(111, autoscale_on=False,
                     xlim=(1, 100), ylim=(-0.1, 0.1))
ax.grid()

line, = ax.plot([], [])#, 'o-', lw=2)
time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)
#energy_text = ax.text(0.02, 0.90, '', transform=ax.transAxes)
line.set_visible(False)
time_text.set_visible(False)
#energy_text.set_visible(False)
def init():
    """initialize animation"""
    line.set_data([], [])
    time_text.set_text('')
    #energy_text.set_text('')
    return line, time_text#, energy_text

def animate(i):
    """perform animation step"""
    if i == 1:
                                        line.set_visible(True)
                                        time_text.set_visible(True)
 #                                       energy_text.set_visible(True)
   # global pendulum, dt
   # pendulum.step(dt)
    dt = iso1d(x1f,rho,rhou,cs,cfl)
    line.set_data(x1c,rhou/rho)
    #time+=dt
    time_text.set_text('time = %.1f' % dt)
    #energy_text.set_text('energy = %.3f J' % pendulum.energy())
    return line, time_text#, energy_text

# choose the interval based on dt and the time to animate one step
#from time import time
#t0 = time()
animate(0)
#t1 = time()
#interval = 1000 * dt - (t1 - t0)

ani = animation.FuncAnimation(fig, animate, frames=300,
                              blit=True, init_func=init)
plt.show()
