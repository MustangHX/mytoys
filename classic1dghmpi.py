import matplotlib
matplotlib.use('macosx')

import matplotlib.pyplot as plt

import numpy as np
import matplotlib.animation as animation
from mpi4py import MPI

gamma=1.4

cfl=0.5

def advect(x1c,x1f,q,u1f,dt,fluxlim,nghost):
  if nghost<1:
    print("ghost zones needs to be at least 1")
    return
  #gradient 
  nx1=len(x1c)
  gr1=np.zeros(nx1+1)
  for i in range(2,nx1-1):
    dq = q[i] - q[i-1]
    if abs(dq) > 0.:
      if u1f[i] >=0.:
        gr1[i] = (q[i-1] - q[i-2])/dq
      else:
        gr1[i] = (q[i+1] - q[i])/dq
  #choose flux limiter
  if True or fluxlim == 'donor-cell':
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
  comm = MPI.COMM_WORLD
  size = comm.Get_size()
  rank = comm.Get_rank()
  if False and rank == 1:
    plt.plot(x1c,q)
    plt.show()
  #print("advect ",rank,nx1,x1c[0],q[0])

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
  elif key_bc == 'open':
    for i in range(nghost):
      rho[i] = rho[nghost]
      rho[nx1-1-i] = rho[nx1-nghost-1]
      rhou[i] = rhou[nghost]
      rhou[nx1-1-i] = rhou[nx1-nghost-1]
    
def hydrostep (x1c, x1f, rho, rhou, cs, cfl, \
    key_bc='open', fluxlim='donor-cell'):
  nghost=2
  nx1 = Np
  comm = MPI.COMM_WORLD
  size = comm.Get_size()
  rank = comm.Get_rank()
  #if rank == 0:
  boundary( rho, rhou, key_bc, nghost)
  #get dt
  left=rank*Np
  right=rank*Np+Np+2*nghost
  #print(left, right, rank)
  dx1=x1f[left+1+nghost:right+1-nghost]-x1f[left+nghost:right-nghost]
  #get dt with cfl
  #print(len(x1c),len(dx1),len(rhou),len(rho))
  dum1 = dx1/(0.5+abs(rhou[left+nghost:right-nghost]/rho[left+nghost:right-nghost]))
  dt1 = cfl * min(dum1)
  dt_m = dt1
  if rank == 1:
    print('absu',rhou[left+nghost:right-nghost])
  dt_m = comm.gather(dt_m, root=0)
  #if rank == 0:
  #  for i in range(size):
  #    assert 

  if rank == 0:
    print(dt_m)
    dt1=min(dt_m)
    print('dt',dt1)
  dt1 = comm.bcast(dt1, root=0)
  dt = dt1
  if rank < size-1:
    comm.Send(rho[right-2*nghost:right-nghost], dest=rank+1,tag=12)
    comm.Recv(rho[right-nghost:right], source=rank+1,tag=13)
  elif rank > 0:
    comm.Recv(rho[left:left+nghost], source=rank-1,tag=12)
    comm.Send(rho[left+nghost:left+2*nghost], dest=rank-1,tag=13)
  if rank < size-1:
    comm.Send(rhou[right-2*nghost:right-nghost], dest=rank+1,tag=14)
    comm.Recv(rhou[right-nghost:right], source=rank+1,tag=15)
  elif rank > 0:
    comm.Recv(rhou[left:left+nghost], source=rank-1,tag=14)
    comm.Send(rhou[left+nghost:left+2*nghost], dest=rank-1,tag=15)
  # get face vel
  u1f=np.zeros(nx1+1+2*nghost)
  for ix1 in range(left,right):
    u1f[ix1-left] = 0.5 * (rhou[ix1]/rho[ix1]\
             +rhou[ix1-1]/rho[ix1-1])

  #advect density
  advect(x1c[left:right],x1f[left:right+1],rho[left:right],u1f,dt,fluxlim,nghost)
  #advect momentum
  #rhoublk=rhou[left:right]

  advect(x1c[left:right],x1f[left:right+1],rhou[left:right],u1f,dt,fluxlim,nghost)
  #do BC again
#  for ix1 in range(left,right):
#    rho[ix1]=rhoblk[ix1-left]
#    rhou[ix1]=rhoublk[ix1-left]
  #if rank == 0:
  boundary( rho, rhou, key_bc, nghost)

  if rank < size-1:
    comm.Send(rho[right-2*nghost:right-nghost], dest=rank+1,tag=16)
    comm.Recv(rho[right-nghost:right], source=rank+1,tag=17)
  elif rank > 0:
    comm.Recv(rho[left:left+nghost], source=rank-1,tag=16)
    comm.Send(rho[left+nghost:left+2*nghost], dest=rank-1,tag=17)
  if rank < size-1:
    comm.Send(rhou[right-2*nghost:right-nghost], dest=rank+1,tag=18)
    comm.Recv(rhou[right-nghost:right], source=rank+1,tag=19)
  elif rank > 0:
    comm.Recv(rhou[left:left+nghost], source=rank-1,tag=18)
    comm.Send(rhou[left+nghost:left+2*nghost], dest=rank-1,tag=19)
  #source term from pressure
  p = rho*cs*cs
  for ix1 in range(left+nghost,right-nghost):
    rhou[ix1] = rhou[ix1] - dt * (p[ix1+1] - p[ix1-1])\
                /(x1c[ix1+1]-x1c[ix1-1])
  #if rank == 0:
  boundary( rho, rhou, key_bc, nghost)
  if rank > 0:
    comm.Send(rho[left:right], dest=0, tag=20)  
  #  comm.Send(rhou[left:right], dest=0, tag=21)
  if rank == 0:
    comm.Recv(rho[left+Np:right+Np], source=1,tag=20)
 # comm.Recv(rhou[left:right], source=1,tag=21)
  return dt

def boundaryad(rho, rhou,rhoe, key_bc, nghost):
  if nghost<1:
    print("ghost zones needs to be at least 1")
    return
  comm = MPI.COMM_WORLD
  size = comm.Get_size()
  rank = comm.Get_rank()

  nx1 = len(rho)
  
  if key_bc == 'periodic':
    for i in range(nghost):

      rho[i] = rho[nx1-2*nghost+i]
      rho[nx1-1-i] = rho[2*nghost-i-1]
      rhou[i] = rhou[nx1-2*nghost+i]
      rhou[nx1-1-i] = rhou[2*nghost-i-1]
      rhoe[i] = rhoe[nx1-2*nghost+i]
      rhoe[nx1-1-i] = rhoe[2*nghost-i-1]
  elif key_bc == 'reflecting':
    for i in range(nghost):
      rho[i] = rho[2*nghost-1-i]
      rho[nx1-1-i] = rho[nx1-2*nghost+i]
      rhou[i] = -rhou[2*nghost-1-i]
      rhou[nx1-1-i] = -rhou[nx1-2*nghost+i]
      rhoe[i] = rhoe[2*nghost-1-i]
      rhoe[nx1-1-i] = rhoe[nx1-2*nghost+i]
  elif key_bc == 'open':
    for i in range(nghost):
      rho[i] = rho[nghost]
      rho[nx1-1-i] = rho[nx1-nghost-1]
      rhou[i] = rhou[nghost]
      rhou[nx1-1-i] = rhou[nx1-nghost-1]
      rhoe[i] = rhoe[nghost]
      rhoe[nx1-1-i] = rhoe[nx1-nghost-1]


def hydrostepad (x1c, x1f, rho, rhou, rhoe, cfl, \
    key_bc='open', fluxlim='donor-cell'):
  nghost=2
  nx1 = Np
  comm = MPI.COMM_WORLD
  size = comm.Get_size()
  rank = comm.Get_rank()
  if rank == 0 or rank == size-1:
    boundaryad( rho, rhou, rhoe, key_bc, nghost)
  #get dt
  left=rank*Np
  right=rank*Np+Np+2*nghost
  #print(left, right, rank)
  dx1=x1f[left+1+nghost:right+1-nghost]-x1f[left+nghost:right-nghost]
  #get dt with cfl
  #print(len(x1c),len(dx1),len(rhou),len(rho))
  dum1 = dx1/(0.5+abs(rhou[left+nghost:right-nghost]/rho[left+nghost:right-nghost]))
  dt1 = cfl * min(dum1)
  dt_m = dt1
  if rank == 1:
    print('absu',rhou[left+nghost:right-nghost])
  dt_m = comm.gather(dt_m, root=0)
  #if rank == 0:
  #  for i in range(size):
  #    assert 
  
  if rank == 0:
    print(dt_m)
    dt1=min(dt_m)
    print('dt',dt1)
  dt1 = comm.bcast(dt1, root=0)
  dt=dt1
  if rank < size-1:
    comm.Send(rho[right-2*nghost:right-nghost], dest=rank+1,tag=1)
    comm.Recv(rho[right-nghost:right], source=rank+1,tag=2)
  elif rank > 0:
    comm.Recv(rho[left:left+nghost], source=rank-1,tag=1)
    comm.Send(rho[left+nghost:left+2*nghost], dest=rank-1,tag=2)
  if rank < size-1:
    comm.Send(rhou[right-2*nghost:right-nghost], dest=rank+1,tag=3)
    comm.Recv(rhou[right-nghost:right], source=rank+1,tag=4)
  elif rank > 0:
    comm.Recv(rhou[left:left+nghost], source=rank-1,tag=3)
    comm.Send(rhou[left+nghost:left+2*nghost], dest=rank-1,tag=4)
  if rank < size-1:
    comm.Send(rhoe[right-2*nghost:right-nghost], dest=rank+1,tag=5)
    comm.Recv(rhoe[right-nghost:right], source=rank+1,tag=6)
  elif rank > 0:
    comm.Recv(rhoe[left:left+nghost], source=rank-1,tag=5)
    comm.Send(rhoe[left+nghost:left+2*nghost], dest=rank-1,tag=6)

  comm.Barrier()

  # get face vel
  u1f=np.zeros(nx1+1+2*nghost)
  for ix1 in range(left,right):
    u1f[ix1-left] = 0.5 * (rhou[ix1]/rho[ix1]\
             +rhou[ix1-1]/rho[ix1-1])
  #advect density
  advect(x1c[left:right],x1f[left:right+1],rho[left:right],u1f,dt,fluxlim,nghost)
  #advect momentum
  advect(x1c[left:right],x1f[left:right+1],rhou[left:right],u1f,dt,fluxlim,nghost)
  #advect total energy
  advect(x1c[left:right],x1f[left:right+1],rhoe[left:right],u1f,dt,fluxlim,nghost)
  #do BC again
  if rank == 0 or rank==size-1:
    boundaryad( rho, rhou, rhoe, key_bc, nghost)
  #source term from pressure

  if rank < size-1:
    comm.Send(rho[right-2*nghost:right-nghost], dest=rank+1,tag=1)
    comm.Recv(rho[right-nghost:right], source=rank+1,tag=2)
  elif rank > 0:
    comm.Recv(rho[left:left+nghost], source=rank-1,tag=1)
    comm.Send(rho[left+nghost:left+2*nghost], dest=rank-1,tag=2)
  if rank < size-1:
    comm.Send(rhou[right-2*nghost:right-nghost], dest=rank+1,tag=3)
    comm.Recv(rhou[right-nghost:right], source=rank+1,tag=4)
  elif rank > 0:
    comm.Recv(rhou[left:left+nghost], source=rank-1,tag=3)
    comm.Send(rhou[left+nghost:left+2*nghost], dest=rank-1,tag=4)
  if rank < size-1:
    comm.Send(rhoe[right-2*nghost:right-nghost], dest=rank+1,tag=5)
    comm.Recv(rhoe[right-nghost:right], source=rank+1,tag=6)
  elif rank > 0:
    comm.Recv(rhoe[left:left+nghost], source=rank-1,tag=5)
    comm.Send(rhoe[left+nghost:left+2*nghost], dest=rank-1,tag=6)

  comm.Barrier()
  u = rhou / rho
  etot = rhoe / rho
  ekin = u*u/2.
  eth  = etot - ekin
  p = (gamma - 1.)*rho*eth
  
  for ix1 in range(left+nghost,right-nghost):
    rhou[ix1] = rhou[ix1] - dt * (p[ix1+1] - p[ix1-1])\
                /(x1c[ix1+1]-x1c[ix1-1])
  if rank == 0 or rank == size-1:  
    boundaryad(rho, rhou, rhoe, key_bc, nghost)
  for ix1 in range(left+nghost,right-nghost):
    rhoe[ix1] = rhoe[ix1] - dt * (p[ix1+1] * u[ix1+1] - p[ix1-1] * u[ix1-1])\
                /(x1c[ix1+1]-x1c[ix1-1]) 
  if rank == 0 or rank == size-1: 
    boundaryad( rho, rhou, rhoe, key_bc, nghost)
  comm.Barrier()
  if rank > 0:
    comm.Send(rho[left:right], dest=0, tag=20)
    comm.Send(rhou[left:right], dest=0, tag=21)
    comm.Send(rhoe[left:right], dest=0, tag=22)
  if rank == 0:
    comm.Recv(rho[left+Np:right+Np], source=1,tag=20)
    comm.Recv(rhou[left+Np:right+Np], source=1,tag=21)
    comm.Recv(rhoe[left+Np:right+Np], source=1,tag=22)
 
  return dt

#initial
N=100
left=0
right=0
rc=0
cs=0.5

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

Np=N/size
print(size,rank)
if size < 2:
  print("Error: number of MPI taska has to be at least 2")


x1f=np.linspace(1,101,N+1)
xmid=0.5*(max(x1f)+min(x1f))
dg=0.1*(max(x1f)-min(x1f))

nghost=2
dx1=x1f[1]-x1f[0]
dx11=x1f[N]-x1f[N-1]
gho_in=np.zeros(nghost)
gho_out=np.zeros(nghost)
for i in range(1,nghost+1):
  gho_in[nghost-i]=x1f[0]-i*dx1
  gho_out[i-1]=x1f[N]+i*dx11

x1f=np.append(gho_in,x1f,axis=0)
x1f=np.append(x1f,gho_out,axis=0)
x1c=(x1f[0:N+2*nghost]+x1f[1:N+1+2*nghost])/2.

print(x1c)

rho=np.zeros(N+2*nghost)
rhou=np.zeros(N+2*nghost)
rhoe=np.zeros(N+2*nghost)



for i in range(N+2*nghost):
  if x1f[i]<=30.:
    rho[i]=1.0
    rhoe[i]=rho[i]*2
  else:
    rho[i]=0.1
    rhoe[i]=rho[i]*1.
#rho=1.+0.3*np.exp(-(x1c-xmid)**2/dg**2)

#if rank == 0:
#  plt.plot(x1c,rhoe/rho-rhou*rhou/rho/rho/2.)
#  plt.show()

count=0

while count<4:
  dt = hydrostepad(x1c, x1f, rho, rhou,rhoe, cfl)
  count+=1
  if True and rank == 0:
    plt.plot(x1c,rho)
    print(count)
    plt.show()

# set up figure and animation
if True or rank == 0:
  fig = plt.figure(figsize=(8,6))
 # if rank>0:
 #   fig.set_visible(False)
  
  ax = fig.add_subplot(321, autoscale_on=False,
                      xlim=(0,101),ylim=(0., 1.2),ylabel='rho')
  ax.grid()

  time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)

  line_rho, = ax.plot([], [])#, 'o-', lw=2)

  ax = fig.add_subplot(323, autoscale_on=False,
                      xlim=(0,101),ylim=(0., 0.9),ylabel='rhou')
  ax.grid()
  line_rhou, = ax.plot([], [])#, 'o-', lw=2)

  ax = fig.add_subplot(325, autoscale_on=False,
                      xlim=(0,101),ylim=(0., 3.5),ylabel='rhoe')
  ax.grid()
  line_rhoe, = ax.plot([], [])#, 'o-', lw=2)

  ax = fig.add_subplot(322, autoscale_on=False,
                      xlim=(0,101),ylim=(0.5, 3.),ylabel='T')
  ax.grid()
  line_T, = ax.plot([], [])#, 'o-', lw=2)

  ax = fig.add_subplot(324, autoscale_on=False,
                      xlim=(0,101),ylim=(0., 9.),ylabel='u')
  ax.grid()
  line_u, = ax.plot([], [])#, 'o-', lw=2)
  ax = fig.add_subplot(326, autoscale_on=False,
                      xlim=(0,101),ylim=(0., 4.),ylabel='p')
  ax.grid()
  line_p, = ax.plot([], [])#, 'o-', lw=2)


  line_rho.set_visible(False)
  line_rhou.set_visible(False)
  line_rhoe.set_visible(False)

  line_T.set_visible(False)
  line_u.set_visible(False)
  line_p.set_visible(False)


  time_text.set_visible(False)
#energy_text.set_visible(False)
else:
  fig=None
def init():
    """initialize animation"""
    line_rho.set_data([], [])
    line_rhou.set_data([], [])
    line_rhoe.set_data([], [])
    line_T.set_data([], [])
    line_u.set_data([], [])
    line_p.set_data([], [])

    time_text.set_text('')
    #energy_text.set_text('')
    return line_rho,line_rhou,line_rhoe,time_text#, energy_text

def animate(i):
    """perform animation step"""
    if i == 1:
      line_rho.set_visible(True)
      line_rhou.set_visible(True)
      line_rhoe.set_visible(True)
      line_T.set_visible(True)
      line_u.set_visible(True)
      line_p.set_visible(True)

      time_text.set_visible(True)
 #                                       energy_text.set_visible(True)
   # global pendulum, dt
   # pendulum.step(dt)
    dt = hydrostepad(x1c, x1f, rho, rhou,rhoe, cfl)
    if True or rank == 0:
      line_rho.set_data(x1c,rho)
      line_rhou.set_data(x1c,rhou)
      line_rhoe.set_data(x1c,rhoe)
   
      eth = rhoe/rho-rhou*rhou/rho/rho/2. 
      line_T.set_data(x1c,eth)
      line_u.set_data(x1c,rhou/rho)
   
      line_p.set_data(x1c,(gamma-1.)*rho*eth)
    #time+=dt
      time_text.set_text('time = %.1f' % dt)
    #energy_text.set_text('energy = %.3f J' % pendulum.energy())
      return line_rho, line_rhou, line_rhoe, line_T, line_u, line_p, time_text#, energy_text

# choose the interval based on dt and the time to animate one step
#from time import time
#t0 = time()
if rank==0:
  animate(0)
#t1 = time()
#interval = 1000 * dt - (t1 - t0)

ani = animation.FuncAnimation(fig, animate, frames=300,
                              blit=True, init_func=init)
plt.show()
