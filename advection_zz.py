#!bin/python

import numpy as np
import matplotlib
matplotlib.use('MACOSX')
import matplotlib.pyplot as plt


def upwind(q,x,u,cfl):#only for evenly spaced x and constant u
  N=len(q)
  dx=x[1]-x[0]
  qtemp=np.zeros(N)
  dt=cfl*dx/u[0]
  for i in range(N):
    qtemp[i]=q[i]
  for i in range(1,N-1):
    if u[i]>0.:
      q[i]=qtemp[i]-dt*u[i]*(qtemp[i]-qtemp[i-1])/dx
    else:
      q[i]=qtemp[i]-dt*u[i]*(qtemp[i+1]-qtemp[i])/dx

  q[0]=qtemp[0]
  return dt


def diffusion(q,x,u,dt,cfl):#explicit diffusion, only for evenly spaced x
  N=len(q)
  diffco=1.0
  dx=x[1]-x[0]
  qtemp=np.zeros(N)
  u=np.zeros(N)
  for i in range(N):
    qtemp[i]=q[i]
  umax=0.
  for i in range(2,N-2):
    #u[i]=diffco*(qtemp[i+1]+qtemp[i-1]-2*qtemp[i])/dx/dx
    dq1=(qtemp[i+1]-qtemp[i])/1/dx
    dq2=(qtemp[i]-qtemp[i-1])/1/dx
    u[i]=diffco*(dq1-dq2)/2/dx
    if abs(u[i])>abs(umax):
      umax=u[i]
  print('max u', umax)
  if (cfl>0.):
    dt=cfl*dx*dx/abs(diffco)
  for i in range(1,N-1):
    q[i]=qtemp[i]+dt*u[i]
  q[0]=qtemp[1]
  q[1]=qtemp[2]
  q[N-1]=qtemp[N-2]
  q[N-2]=qtemp[N-3]
  return dt

def diffusion_backeuler(q,x,u,dt,cfl):#implicit diffusion
  N=len(q)
  diffco=1.
# dx=x[1]-x[0]
  qtemp=np.zeros(N)
  u=np.zeros(N)
  # data for tridiagonal matrix
  A = np.zeros((N,N))
  b = np.zeros(N)
  for i in range(N):
    qtemp[i]=q[i]
  umax=0.
  for i in range(1,N-1):
    dx=x[i]-x[i-1]
    u[i]=diffco*(qtemp[i+1]+qtemp[i-1]-2*qtemp[i])/dx/dx
    if abs(u[i])>abs(umax):
      umax=u[i]
  print('max u', umax)
  if(cfl>0.):
    dt=cfl*dx*dx/abs(diffco)
#F=diffco*dt/dx/dx
  for i in range(1,N-1):
    dx=x[i]-x[i-1]
    F=diffco*dt/dx/dx
    A[i,i-1] = -F
    A[i,i+1] = -F
    A[i,i] = 1 + 2*F
  A[0,0] = A[N-1,N-1] = 1.
 # Thomas Algorithm
  c=np.zeros(N)
  d=np.zeros(N)
  c[0]=A[0,1]/A[0,0]
  for i in range(1,N-1):
    c[i]=A[i,i+1]/(A[i,i]-A[i,i-1]*c[i-1])
  d[0]=qtemp[0]/A[0,0]
  for i in range(1,N-1):
    d[i] = (qtemp[i]-A[i,i-1]*d[i-1])/\
           (A[i,i]-A[i,i-1]*c[i-1])
  q[N-1]=d[N-1]
  for i in range(N-2,0,-1):
    q[i] = d[i]-c[i]*q[i+1]
  return dt


def peak_initial(q,x):
  for i in range(len(x)):
    if x[i]<40.:
      q[i]=0.
    elif x[i]>=40 and x[i]<=60:
      q[i]=1.
    else:
      q[i]=0.
  return 0

def u_intial(u,x):
  for i in range(len(x)):
    u[i]=1

  return 0

x=np.linspace(0,100,100)
u=np.zeros(100)



q=np.zeros(len(x))

peak_initial(q,x)
u_intial(u,x)
plt.plot(x,q)
plt.ylim(-0.01,1.1)
plt.show()

count=0
dt=0.3
while count<100:
  
  cfl=-0.3 # use negative value to use fixed dt
  #dt=upwind(q,x,u,cfl)
  print(dt)
  dt=diffusion(q,x,u,dt,cfl)
  count+=1
  plt.title('t='+str(count*dt))
  plt.plot(x,q)
  plt.ylim(-0.01,1.1)
  plt.show()
