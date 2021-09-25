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
  #print(len(dx1),len(rhou1),len(rho))
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

def advect(x1c,x1f,q,u1f,dt,fluxlim,nghost):
  if nghost<1:
    print("ghost zones needs to be at least 1")
    return
  #gradient 
  nx1=len(x1c)
  gr1=np.zeros(nx1+1)
  for i in range(2,nx1-1):
    dq = q[i] - q[i-1]
    if abs(dq) >= 0.:
      gr1[i] = (q[i-1] - q[i-2])/dq
    else:
      gr1[i] = (q[i+1] - q[i])/dq
  #choose flux limiter
  if fluxlim == 'donor-cell':
    phi = np.zeros(nx1+1)
  elif fluxlim == 'superbee':
    phi = np.zeros(nx1+1)
    for i in range(1,nx1):
      a = min([1.,2.*gr1[i]])
      b = min([2.,gr1[i]])
      phi[i] = max([0.,a,b])
  #construct the flux
  flux1 = np.zeros(nx1+1)
  for i in range(1,nx1):
    if u1f[i] >= 0. :
      flux1[i] = u1f[i] * q[i-1]
    else:
      flux1[i] = u1f[i] * q[i]

    flux1[i] = flux1[i] + 0.5 * abs( u1f[i] ) *\
              (1. - abs(u1f[i]*dt/(x1c[i]-x1c[i-1]))) * \
              phi[i] * (q[i]-q[i-1])
  #update active zones
  for i in range(nghost,nx1-nghost):
    q[i] = q[i] - dt * ( flux1[i+1]-flux1[i] ) / (x1f[i+1]-x1f[i])

#boundary conditions

def boundary ( rho, rhou, key_bc, nghost):
  if nghost<1:
    print("ghost zones needs to be at least 1")
    return

  nx1 = len(rho)

  if key_bc == 'periodic':
    for i in range(nghost):
      rho[i] = rho[nx1-2*nghost+i]
      rho[nx1-1-i] = rho[2*nghost-i-1]
      rhou[i] = rhou[nx1-2*nghost+i]
      rhou[nx1-1-i] = rhou[2*nghost-i-1]
      
  elif key_bc == 'reflecting':
    for i in range(nghost):
      rho[i] = rho[2*nghost-1-i]
      rho[nx1-1-i] = rho[nx1-2*nghost+i]
      rhou[i] = -rhou[2*nghost-1-i]
      rhou[nx1-1-i] = -rhou[nx1-2*nghost+i]
    
def hydrostep2d (x1c, x1f, x2c, x2f, rho, rhou1, rhou2, cs, cfl, \
    key_bc1='reflecting', key_bc2='reflecting',fluxlim='donor-cell'):
  nghost=2
  nx1 = len(x1c)
  nx2 = len(x2c)
  #BC at x1 boundary
  for ix2 in range(nghost,nx2-nghost):
    boundary( rho[ix2], rhou1[ix2], key_bc1, nghost)
    boundary( rho[ix2], rhou2[ix2], key_bc1, nghost)
  #BC ar x2 boundary
  for ix1 in range(nghost,nx1-nghost):
    boundary( rho[:,ix1], rhou1[:,ix1], key_bc2, nghost)
    boundary( rho[:,ix1], rhou2[:,ix1], key_bc2, nghost)

  #get dt
  dx1=x1f[1:nx1+1]-x1f[0:nx1]
  dx2=x2f[1:nx2+1]-x2f[0:nx2]
  
  #get dt with cfl
  #print(len(x1c),len(dx1),len(rhou1),len(rho))
  dum1 = dx1/(cs+abs(rhou1/rho))
  #print(np.shape(dum1))
  dt1 = cfl * np.min(dum1)
  dt = dt1
  # get face vel
  u1f=np.zeros((nx2+1,nx1+1))
  u2f=np.zeros((nx2+1,nx1+1))

  for ix2 in range(1,nx2):
    for ix1 in range(1,nx1):
      u1f[ix2,ix1] = 0.5 * (rhou1[ix2,ix1]/rho[ix2,ix1]\
             +rhou1[ix2,ix1-1]/rho[ix2,ix1-1])
  for ix2 in range(1,nx2):
    for ix1 in range(1,nx1):
      u2f[ix2,ix1] = 0.5 * (rhou2[ix2,ix1]/rho[ix2,ix1]\
             +rhou2[ix2-1,ix1]/rho[ix2-1,ix1])

  #advect density for x1
  for ix2 in range(nghost,nx2-nghost):
    advect(x1c,x1f,rho[ix2],u1f[ix2],dt,fluxlim,nghost)
    #advect rhou2
    advect(x1c,x1f,rhou2[ix2],u1f[ix2],dt,fluxlim,nghost)
    #advect momentum
    advect(x1c,x1f,rhou1[ix2],u1f[ix2],dt,fluxlim,nghost)
  #do BC again
  #BC at x1 boundary
  for ix2 in range(nghost,nx2-nghost):
    boundary( rho[ix2], rhou1[ix2], key_bc1, nghost)
    boundary( rho[ix2], rhou2[ix2], key_bc1, nghost)
  #BC ar x2 boundary
  for ix1 in range(nghost,nx1-nghost):
    boundary( rho[:,ix1], rhou1[:,ix1], key_bc2, nghost)
    boundary( rho[:,ix1], rhou2[:,ix1], key_bc2, nghost)
  #source term from pressure
  p = rho*cs*cs
  for ix2 in range(nghost,nx2-nghost):
    for ix1 in range(nghost,nx1-nghost):
      rhou1[ix2,ix1] = rhou1[ix2,ix1] - dt * (p[ix2,ix1+1] - p[ix2,ix1-1])\
                /(x1c[ix1+1]-x1c[ix1-1])
  #BC at x1 boundary
  for ix2 in range(nghost,nx2-nghost):
    boundary( rho[ix2], rhou1[ix2], key_bc1, nghost)
    boundary( rho[ix2], rhou2[ix2], key_bc1, nghost)
  #BC ar x2 boundary
  for ix1 in range(nghost,nx1-nghost):
    boundary( rho[:,ix1], rhou1[:,ix1], key_bc2, nghost)
    boundary( rho[:,ix1], rhou2[:,ix1], key_bc2, nghost)
  #advect density for x2
  for ix1 in range(nghost,nx1-nghost):
    advect(x2c,x2f,rho[:,ix1],u2f[:,ix1],dt,fluxlim,nghost)
    #advect rhou1
    advect(x2c,x2f,rhou1[:,ix1],u2f[:,ix1],dt,fluxlim,nghost)
    #advect momentum
    advect(x2c,x2f,rhou2[:,ix1],u2f[:,ix1],dt,fluxlim,nghost)
  #do BC again
  #BC at x1 boundary
  for ix2 in range(nghost,nx2-nghost):
    boundary( rho[ix2], rhou1[ix2], key_bc1, nghost)
    boundary( rho[ix2], rhou2[ix2], key_bc1, nghost)
  #BC ar x2 boundary
  for ix1 in range(nghost,nx1-nghost):
    boundary( rho[:,ix1], rhou1[:,ix1], key_bc2, nghost)
    boundary( rho[:,ix1], rhou2[:,ix1], key_bc2, nghost)
  #source term from pressure
  p = rho*cs*cs
  for ix2 in range(nghost,nx2-nghost):
    for ix1 in range(nghost,nx1-nghost):
      rhou2[ix2,ix1] = rhou2[ix2,ix1] - dt * (p[ix2+1,ix1] - p[ix2-1,ix1])\
                /(x2c[ix2+1]-x2c[ix2-1])
    
  
  return dt
#initial
N1=70
N2=140

x1f=np.linspace(1,101,N1+1)
x1mid=0.5*(max(x1f)+min(x1f))
dg1=0.1*(max(x1f)-min(x1f))

x2f=np.linspace(1,201,N2+1)
x2mid=0.5*(max(x2f)+min(x2f))
dg2=0.1*(max(x2f)-min(x2f))

nghost=2

dx1=x1f[1]-x1f[0]
dx11=x1f[N1]-x1f[N1-1]
gho_in=np.zeros(nghost)
gho_out=np.zeros(nghost)
for i in range(1,nghost+1):
  gho_in[nghost-i]=x1f[0]-i*dx1
  gho_out[i-1]=x1f[N1]+i*dx11

x1f=np.append(gho_in,x1f,axis=0)
x1f=np.append(x1f,gho_out,axis=0)
x1c=(x1f[0:N1+2*nghost]+x1f[1:N1+1+2*nghost])/2.

print(x1c)

dx2=x2f[1]-x2f[0]
dx21=x2f[N2]-x2f[N2-1]
gho_in=np.zeros(nghost)
gho_out=np.zeros(nghost)
for i in range(1,nghost+1):
  gho_in[nghost-i]=x2f[0]-i*dx1
  gho_out[i-1]=x2f[N2]+i*dx11

x2f=np.append(gho_in,x2f,axis=0)
x2f=np.append(x2f,gho_out,axis=0)
x2c=(x2f[0:N2+2*nghost]+x2f[1:N2+1+2*nghost])/2.

rho=np.zeros((N2+2*nghost,N1+2*nghost))
rhou1=np.zeros((N2+2*nghost,N1+2*nghost))
rhou2=np.zeros((N2+2*nghost,N1+2*nghost))



for ix2 in range(N2+2*nghost):
  for ix1 in range(N1+2*nghost):
  #  if x1c[ix1]>30. and x1c[ix1]<70. and x2c[ix2]>-1. and x2c[ix2]<30.:
    #if x1f[ix1]>50.:
  #    rhou2[ix2,ix1]=5
  #    rho[ix2,ix1]=15
  #  else:
      #rhou1[ix2,ix1]=0.5
  #    rho[ix2,ix1]=0.5
    rho[ix2,ix1]=1.+0.5*np.exp(-((x1c[ix1]-x1mid)**2+(x2c[ix2]-x2mid)**2)/(dg1**2+dg2**2))
cs=0.1


count=0
fig=plt.figure()
ax=fig.add_subplot(111)
im=ax.pcolormesh(x1c,x2c,rho)#,vmin=vmin_list[j],vmax=vmax_list[j])
ax.set_aspect('equal')
plt.show()
time=0.
while count<1:

  cfl=0.5
  dt=hydrostep2d(x1c, x1f, x2c, x2f, rho, rhou1, rhou2, cs, cfl)
  time+=dt
  print(time)
  if count>8:
    im=ax.pcolormesh(x1c,x2c,rho)#,vmin=vmin_list[j],vmax=vmax_list[j])
    ax.set_aspect('equal')
    plt.show()
  count+=1


# set up figure and animation
fig = plt.figure()
ax = fig.add_subplot(111,aspect='equal', autoscale_on=False,
                      xlim=(x1f[0],x1f[N1+2*nghost]),ylim=(min(x1f), max(x2f)))
#ax.grid()

xgrid,ygrid=np.meshgrid(x1c,x2c)
mesh=ax.pcolormesh(xgrid,ygrid,rho)#, 'o-', lw=2)
time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)
#energy_text = ax.text(0.02, 0.90, '', transform=ax.transAxes)
mesh.set_visible(False)
time_text.set_visible(False)
#energy_text.set_visible(False)
def init():
    """initialize animation"""
    mesh.set_array([])
    time_text.set_text('')
    #time=0.
    #energy_text.set_text('')
    return mesh, time_text#, energy_text

def animate(i):
    """perform animation step"""
    if i == 1:
                                        mesh.set_visible(True)
                                        time_text.set_visible(True)
 #                                       energy_text.set_visible(True)
   # global pendulum, dt
   # pendulum.step(dt)
    dt = hydrostep2d(x1c, x1f, x2c, x2f, rho, rhou1, rhou2, cs, cfl)
   # necessary for shading='flat'
    rhoplot = rho[:-1, :-1]
    mesh.set_array(rhoplot.ravel())
    #time+=dt
    time_text.set_text('time = %.1f' % dt)
    #energy_text.set_text('energy = %.3f J' % pendulum.energy())
    return mesh, time_text#, energy_text

# choose the interval based on dt and the time to animate one step
#from time import time
#t0 = time()
animate(0)
#t1 = time()
#interval = 1000 * dt - (t1 - t0)

ani = animation.FuncAnimation(fig, animate, frames=300,
                              blit=True, init_func=init)
plt.show()
