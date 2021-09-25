#!bin/python

import numpy as np

import matplotlib
matplotlib.use('macosx')

import matplotlib.pyplot as plt

def centerdiff(q,x,u,cfl):#only for evenly spaced x
  N=len(q)
  dx=x[1]-x[0]
  qtemp=np.zeros(N)
  dt=cfl*dx/u
  for i in range(N):
    qtemp[i]=q[i]
  for i in range(1,N-1):
    q[i]=qtemp[i]-dt*u*(qtemp[i+1]-qtemp[i-1])/2/dx
  q[0]=qtemp[0]-dt*u*(qtemp[1]-qtemp[0])/dx
  q[N-1]=qtemp[N-1]-dt*u*(qtemp[N-1]-qtemp[N-2])/dx
  return 0

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

def upwind_cons(q,x,u,cfl):#only for evenly spaced x 
  N=len(q)
  dx=x[1]-x[0]
  qtemp=np.zeros(N)
  dt=cfl*dx/u[0]
  for i in range(N):
    qtemp[i]=q[i]
  for i in range(1,N-1):
    if u[i]>0.:
      q[i]=qtemp[i]-dt*(u[i]*qtemp[i]-u[i-1]*qtemp[i-1])/dx
    else:
      q[i]=qtemp[i]-dt*(u[i+1]*qtemp[i+1]-u[i]*qtemp[i])/dx

  q[0]=qtemp[0]
  return dt

def van_leer(q,x,u,cfl):
  N=len(q)
  dx=x[1]-x[0]
  qtemp=np.zeros(N)
  dt=cfl*dx/u[0]
  for i in range(N):
    qtemp[i]=q[i]
  for i in range(2,N-2):
    dxi=x[i+1]-x[i]
    dxi01=x[i]-x[i-1] # 0 means minus
    dxi02=x[i-1]-x[i-2]
    dxi1=x[i+2]-x[i+1]
    if u[i]>0.:
      if (qtemp[i-1]-qtemp[i-2])/dxi02*\
         (qtemp[i]-qtemp[i-1])/dxi01>0.:
        dqi01=2.*(qtemp[i-1]-qtemp[i-2])/dxi02*\
              (qtemp[i]-qtemp[i-1])/dxi01\
              /((qtemp[i-1]-qtemp[i-2])/dxi02\
              +(qtemp[i]-qtemp[i-1])/dxi01)
      else:
        dqi01=0.
#     dqi01=0.
      qface01=qtemp[i-1]+(dxi01-u[i]*dt)*dqi01/2.

      if (qtemp[i]-qtemp[i-1])/dxi01*\
         (qtemp[i+1]-qtemp[i])/dxi>0.:
        dqi=2.*(qtemp[i]-qtemp[i-1])/dxi01*\
              (qtemp[i+1]-qtemp[i])/dxi\
              /((qtemp[i]-qtemp[i-1])/dxi01\
                  +(qtemp[i+1]-qtemp[i])/dxi)
      else:
        dqi=0.
#     dqi=0.
      qface=qtemp[i]+(dxi-u[i]*dt)*dqi/2.
#      q[i]=qtemp[i]+(qface01*u[i-1]-qface*u[i])*dt/dxi01
    else:
      if (qtemp[i]-qtemp[i-1])/dxi01*\
         (qtemp[i+1]-qtemp[i])/dxi>0.:
        dqi=2.*(qtemp[i]-qtemp[i-1])/dxi01*\
              (qtemp[i+1]-qtemp[i])/dxi\
              /((qtemp[i]-qtemp[i-1])/dxi01\
                +(qtemp[i+1]-qtemp[i])/dxi)
      else:
        dqi=0.
#     dqi=0.
      qface01=qtemp[i]-(dxi-u[i]*dt)*dqi/2.
      
      if (qtemp[i+1]-qtemp[i])/dxi*\
         (qtemp[i+2]-qtemp[i+1])/dxi1>0.:
        dqi1=2.*(qtemp[i+1]-qtemp[i])/dxi*\
              (qtemp[i+2]-qtemp[i+1])/dxi1\
              /((qtemp[i+1]-qtemp[i])/dxi\
                +(qtemp[i+2]-qtemp[i+1])/dxi1)
      else:
        dqi1=0.
#dqi1=0.
      qface=qtemp[i+1]-(dxi1-u[i+1]*dt)*dqi1/2.
#     q[i]=qtemp[i]+(qface01*u[i]-qface*u[i+1])*dt/dxi
    q[i]=qtemp[i]+(qface01*u[i-1]-qface*u[i])*dt/dxi01

  return dt


def diffusion(q,x,u,dt):#only for evenly spaced x
  N=len(q)
  diffco=0.1
  dx=x[1]-x[0]
  qtemp=np.zeros(N)
  u=np.zeros(N)
  for i in range(N):
    qtemp[i]=q[i]
  umax=0.
  for i in range(1,N-1):
    u[i]=diffco*(qtemp[i+1]+qtemp[i-1]-2*qtemp[i])/dx/dx
#dq1=(qtemp[i+2]-qtemp[i])/2/dx
#   dq2=(qtemp[i]-qtemp[i-2])/2/dx

#u[i]=diffco*(dq1-dq2)/2/dx
    if abs(u[i])>abs(umax):
      umax=u[i]
  print('max u', umax)
#dt=cfl*dx/abs(umax)
  for i in range(1,N-1):
    q[i]=qtemp[i]+dt*u[i]
  q[0]=qtemp[0]
  q[N-1]=qtemp[N-1]
  return 0

def diffusion_backeuler(q,x,u,dt):#only for evenly spaced x
  N=len(q)
  diffco=20.
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
#dt=cfl*dx/abs(umax)
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
  return 0

def step_initial(q,x):
  for i in range(len(x)):
    if x[i]<30.:
      q[i]=1.
    else:
      q[i]=0.
  return 0

def peak_initial(q,x):
  for i in range(len(x)):
    if x[i]<30.:
      q[i]=0.
    elif x[i]>=30 and x[i]<=40:
      q[i]=1.
    elif x[i]>60 and x[i]<70:
      q[i]=1.
    else:
      q[i]=0.
  return 0

def u_intial(u,x):
  for i in range(len(x)):
    if i<len(x)/2.:
      u[i]=10
    elif False and i==len(x)/2:
      u[i]=0.
    else:
      u[i]=-10

  return 0

x=np.linspace(0,100,100)
u=np.linspace(0,100,100)


#for i in range(100):
#  x[i]=1.0*(100./1.)**(i*1./100.)

q=np.zeros(len(x))

peak_initial(q,x)
u_intial(u,x)
plt.plot(x,q)
plt.show()

count=0
while count<100:
  
  cfl=0.3
#  centerdiff(q,x,u,cfl)
#  dt=upwind(q,x,u,cfl)
  dt=van_leer(q,x,u,cfl)
#  diffusion_backeuler(q,x,u,dt)
  plt.plot(x,q)
  plt.show()
