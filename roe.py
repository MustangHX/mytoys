import matplotlib
matplotlib.use('macosx')

import matplotlib.pyplot as plt

import numpy as np
import matplotlib.animation as animation

gamma=1.4

cfl=0.5

RECONS=True

#global time 0
def func_flux(q,gamma):
  # Primary properties
  r=q[0];
  u=q[1]/r;
  E=q[2]/r;
  p=(gamma-1.)*r*(E-0.5*u**2);

  # Flux vector of conserved properties
  F0 = np.array(r*u)
  F1 = np.array(r*u**2+p)
  F2 = np.array(u*(r*E+p))
  flux=np.array([ F0, F1, F2 ]);

  return (flux)

def func_prim2cons(r,u,p,gamma):
  q0=r;
  q1=r*u;
  q2=p/(gamma-1.)+0.5*r*u**2;
  q =np.array([ q0, q1, q2 ]);
  
  return (q)

def func_cons2prim(q,gamma):
  r=q[0];
  u=q[1]/r;
  E=q[2]/r;
  p=(gamma-1.)*r*(E-0.5*u**2);

  return (r,u,p)

def minmod(a, b):
  if a*b<=0:
    return 0.
  elif abs(a)<abs(b):
    return a
  else:
    return b

def reconstruct(x1c,x1f,q,key_m='minmod'):
  nx1=len(x1c)
  slp=np.zeros(nx1)
  qfl=np.zeros(nx1)
  qfr=np.zeros(nx1)
  dx=x1f[1:]-x1f[0:-1]
  if key_m=='superbee':
    for i in range(nghost-1,nx1-nghost+1):
      slp1=minmod((q[i+1]-q[i])/dx[i+1],2*(q[i]-q[i-1])/dx[i])
      slp2=minmod(2*(q[i+1]-q[i])/dx[i+1],(q[i]-q[i-1])/dx[i])
      if abs(slp1) > abs(slp2):
        slp[i]=slp1
      else:
        slp[i]=slp2
      qfr[i]=q[i]-slp[i]*dx[i]*0.5
      qfl[i+1]=q[i]+slp[i]*dx[i]*0.5
    qf=np.array([ qfl[nghost:-nghost+1], qfr[nghost:-nghost+1] ]);
  elif key_m=='donorcell':
    for i in range(nghost-1,nx1-nghost+1):
      qfr[i]=q[i]
      qfl[i]=q[i-1]
    qf=np.array([ qfl[nghost:-nghost+1], qfr[nghost:-nghost+1] ]);
#    qf=np.array([q[nghost-1:-nghost],q[nghost:-nghost+1]])
  elif key_m == 'minmod':
    for i in range(nghost-1,nx1-nghost+1):
      slp[i]=minmod((q[i+1]-q[i])/dx[i+1],(q[i]-q[i-1])/dx[i])
      print("slp",(q[i+1]-q[i])/dx[i+1],(q[i]-q[i-1])/dx[i],slp[i])
      qfr[i]=q[i]-slp[i]*dx[i]*0.5
      qfl[i+1]=q[i]+slp[i]*dx[i]*0.5
    qf=np.array([ qfl[nghost:-nghost+1], qfr[nghost:-nghost+1] ]);
 
  print("qf shape",np.shape(qf))
  return qf
      

def roestepBAK(x1c, x1f, rho, rhou, rhoe, cfl,\
    key_bc='open', fluxlim='donor-cell'):
  nghost=2
  # Compute enthalpy
  nx=len(x1f)-2*nghost
  q=np.array([rho[nghost-1:-nghost], rhou[nghost-1:-nghost], rhoe[nghost-1:-nghost]])
  (r,u,p) = func_cons2prim(q,gamma)
  htot = gamma/(gamma-1)*p/r+0.5*u**2;
  dx1=x1f[nghost:nx+nghost]-x1f[nghost-1:nx+nghost-1]
  cs=np.sqrt(gamma*p/r)
 
  #get dt with cfl
  print(len(x1c),len(dx1),len(rhou),len(rho))
  dum1 = dx1/(cs+abs(u))
  dt1 = cfl * min(dum1)
  dt = dt1

  Phi=np.zeros((3,nx-1))
  #=============================================
  # Compute Roe averages
  #=============================================
  for j in range (0,nx-1):

    R=np.sqrt(r[j+1]/r[j]);                          # R_{j+1/2}
    rmoy=R*r[j];                                  # {hat rho}_{j+1/2}
    umoy=(R*u[j+1]+u[j])/(R+1);                   # {hat U}_{j+1/2}
    hmoy=(R*htot[j+1]+htot[j])/(R+1);             # {hat H}_{j+1/2}
    amoy=np.sqrt((gamma-1.0)*(hmoy-0.5*umoy*umoy));  # {hat a}_{j+1/2}

    # Auxiliary variables used to compute P_{j+1/2}^{-1}
    alph1=(gamma-1)*umoy*umoy/(2*amoy*amoy);
    alph2=(gamma-1)/(amoy*amoy);

    #=============================================
    # Compute Roe matrix  |A_{j+1/2}| (W_{j+1}-W_j)
    #=============================================
    # Compute vector (W_{j+1}-W_j)
    wdif = q[:,j+1]-q[:,j];

    # Compute matrix P^{-1}_{j+1/2}
    Pinv = np.array([[0.5*(alph1+umoy/amoy), -0.5*(alph2*umoy+1/amoy),  alph2/2],
                        [1-alph1,                alph2*umoy,                -alph2 ],
                        [0.5*(alph1-umoy/amoy),  -0.5*(alph2*umoy-1/amoy),  alph2/2]]);

    # Compute matrix P_{j+1/2}
    P    = np.array([[ 1,              1,              1              ],
                        [umoy-amoy,        umoy,           umoy+amoy      ],
                        [hmoy-amoy*umoy,   0.5*umoy*umoy,  hmoy+amoy*umoy ]]);

    # Compute matrix Lambda_{j+1/2}
    lamb = np.array([[ abs(umoy-amoy),  0,              0                 ],
                        [0,                 abs(umoy),      0                 ],
                        [0,                 0,              abs(umoy+amoy)    ]]);
    # Compute flux
    A=np.dot(P,lamb)
    A=np.dot(A,Pinv)
    Phi[:,j]=np.dot(A,wdif)

  #=============================================
  # Compute  Phi=(F(W_{j+1}+F(W_j))/2-|A_{j+1/2}| (W_{j+1}-W_j)/2
  #=============================================
  F = func_flux(q,gamma);
  Phi=0.5*(F[:,0:nx-1]+F[:,1:nx])-0.5*Phi
  q0=q
  dF = (Phi[:,1:-1]-Phi[:,0:-2])
  #print(np.shape(dF[0]),np.shape(rho[nghost:-nghost]))
  q[:,1:-2] = q0[:,1:-2]-dt/dx1[0]*dF
  rho[nghost:-nghost-2]=rho[nghost:-nghost-2]-dt/dx1[0]*dF[0]
  rhou[nghost:-nghost-2]=rhou[nghost:-nghost-2]-dt/dx1[0]*dF[1]
  rhoe[nghost:-nghost-2]=rhoe[nghost:-nghost-2]-dt/dx1[0]*dF[2]
  #print('dF',dF)
#  print('dF',np.shape(dF),'Phi',np.shape(Phi),'F',np.shape(F))

  q[:,0]=q0[:,0]; q[:,-1]=q0[:,-1]; # Neumann BCs
  #rho=q[0]
  #rhou=q[1]
  #rhoe=q[2]
  return (dt)

def roestepf(x1c, x1f, rhof, rhouf, rhoef):
  
  gam1=gamma-1.
  nx1 = len(x1c)
  rhol = np.zeros(nx1+1)
  ul = np.zeros(nx1+1)
  htotl = np.zeros(nx1+1)
  rhor = np.zeros(nx1+1)
  ur = np.zeros(nx1+1)
  htotr = np.zeros(nx1+1)
  phi = np.zeros((3,nx1+1))
  uf=rhouf/rhof
  rhol = rhof[0]
  #print('rhof shape',np.shape(rhof))
  rhor = rhof[1]
  ul   = rhouf[0]/rhol
  ur   = rhouf[1]/rhor
  ethl=rhoef[0]/rhol-0.5*ul*ul
  ethr=rhoef[1]/rhor-0.5*ur*ur

  pl=gam1*rhol*ethl
  pr=gam1*rhor*ethr
  htotl=rhoef[0]/rhol+pl/rhol
  htotr=rhoef[1]/rhor+pr/rhor




  # get flux
  Frhol = np.array(rhol*ul)
  Frhor = np.array(rhor*ur)

  Frhoul = np.array(rhol*ul*ul+pl)
  Frhour = np.array(rhor*ur*ur+pr)


  
  Frhoel = np.array(ul*(rhoef[0]+pl))
  Frhoer = np.array(ur*(rhoef[1]+pr))

  fluxl = np.array([Frhol, Frhoul, Frhoel])
  fluxr = np.array([Frhor, Frhour, Frhoer])

  #print('phi flux',np.shape(flux),flux)
  for ix1 in range(0,nx1+1):
#    rhol[ix1]=rhof[0,ix1]
#    ul[ix1]=rhouf[0,ix1]/rhof[0,ix1]
#    ethl[ix1]=rhoef[0,ix1]/rhol[ix1]-0.5*ul[ix1]*ul[ix1]
#    pl[ix1]=gam1*rhol[ix1]*ethl[ix1]
#    htotl[ix1]=rhoef[0,ix1]/rhol[ix1]+pl[ix1]/rhol[ix1]

#    rhor[ix1]=rhof[1,ix1]
#    ur[ix1]=rhouf[1,ix1]/rhof[1,ix1]
#    ethr[ix1]=rhoef[1,ix1]/rhor[ix1]-0.5*ur[ix1]*ur[ix1]
#    pr[ix1]=gam1*rhor[ix1]*ethr[ix1]
#    htotr[ix1]=rhoef[1,ix1]/rhor[ix1]+pr[ix1]/rhor[ix1]


    # get Roe average
    sqrtrhol=np.sqrt(rhol[ix1])
    sqrtrhor=np.sqrt(rhor[ix1])
    isrholr=sqrtrhol+sqrtrhor
    rhoroe=sqrtrhol*sqrtrhor
    uroe=(sqrtrhol*ul[ix1]+sqrtrhor*ur[ix1])/isrholr
    htotroe=(sqrtrhol*htotl[ix1]+sqrtrhor*htotr[ix1])/isrholr
    csroe=np.sqrt((gam1)*(htotroe-0.5*uroe*uroe))
    csroe2=csroe*csroe
    uroe2=uroe*uroe
    #print('gam1')
    # get eigenvalues eigen vectors in [rhoe, rhou, rho]
    # right eigen vectors or P
    P = np.array([[1.,                 1.,                            1.],
                [uroe-csroe,         uroe,                  uroe+csroe],
                [htotroe-csroe*uroe, 0.5*uroe*uroe, htotroe+csroe*uroe]])
  #left eigenvector or P^-1
    na = 0.5/csroe/csroe
    Pinv=np.array([[na*(gam1*uroe2/2.+uroe*csroe), -na*(gam1*uroe+csroe), na*gam1],
                 [1.-na*gam1*uroe2,              gam1*uroe/csroe2,  -gam1/csroe2],
                 [na*(gam1*uroe2/2.-uroe*csroe), -na*(gam1*uroe-csroe),   na*gam1]])
    # eigenvalues matrix or diagonized Jacobian
    lamb = np.array([[ abs(uroe-csroe), 0,             0              ],
                     [0,                abs(uroe),     0              ],
                     [0,                0,             abs(uroe+csroe)]])
    #cons difference at each side of face
    dU = np.array([rhor[ix1]-rhol[ix1],rhouf[1,ix1]-rhouf[0,ix1], rhoef[1,ix1]-rhoef[0,ix1]])
    #get dU projection on eigenvectors (matrix A)
    A = np.dot(P,lamb)
    A = np.dot(A,Pinv)
    phi[:,ix1] = np.dot(A,dU)
    #print(ix1,x1c[ix1], htotr[ix1],p[ix1],0.5*ur[ix1]*ur[ix1])
    # to be added: check negative in intermediate states
#    print(ix1,'htot',htotroe,rhoef[0,ix1],rhoef[1,ix1],pl[ix1],pr[ix1],rhol[ix1],rhor[ix1])
#    print(ix1,Pinv)
  #compute Roe flux
  phiroe = 0.5*(fluxl+fluxr)-0.5*phi

  dF = phiroe[:,1:]-phiroe[:,0:-1]
  #rho[nghost:nx1-nghost] = rho[nghost:nx1-nghost] - dt*dF[0,:]/dx1[nghost:-nghost]
  #rhou[nghost:nx1-nghost] = rhou[nghost:nx1-nghost] - dt*dF[1,:]/dx1[nghost:-nghost]
  #rhoe[nghost:nx1-nghost] = rhoe[nghost:nx1-nghost] - dt*dF[2,:]/dx1[nghost:-nghost]
  #print('fluxl',np.shape(fluxl))

  #print('fluxl',fluxl)


  return dF

 
def roestep(x1c, x1f, rho, rhou, rhoe, cfl, \
    key_bc='open', fluxlim='donor-cell'):
  nghost=2
  gam1=gamma-1.
  nx1 = len(x1c)
  rhol = np.zeros(nx1+1)
  ul = np.zeros(nx1+1)
  htotl = np.zeros(nx1+1)
  rhor = np.zeros(nx1+1)
  ur = np.zeros(nx1+1)
  htotr = np.zeros(nx1+1)
  phi = np.zeros((3,nx1+1))


  u=rhou/rho
  eth=rhoe/rho-0.5*u*u
  p=gam1*rho*eth
  cs=np.sqrt(gamma*p/rho)
    #get dt
  dx1=x1f[1:nx1+1]-x1f[0:nx1]
  #get dt with cfl
  #print(len(x1c),len(dx1),len(rhou),len(rho))
  dum1 = dx1/(cs+abs(rhou/rho))
  dt1 = cfl * min(dum1)
  dt = dt1

  # get flux
  Frho = np.array(rho[nghost-1:nx1-nghost+1]*u[nghost-1:nx1-nghost+1])
  Frhou = np.array(rho[nghost-1:nx1-nghost+1]*u[nghost-1:nx1-nghost+1]*\
          u[nghost-1:nx1-nghost+1]+p[nghost-1:nx1-nghost+1])
  Frhoe = np.array(u[nghost-1:nx1-nghost+1]*(rhoe[nghost-1:nx1-nghost+1]+\
          p[nghost-1:nx1-nghost+1]))
  flux = np.array([Frho, Frhou, Frhoe])
  #print('phi flux',np.shape(flux),flux)
  for ix1 in range(nghost,nx1-nghost+1):
    rhol[ix1]=rho[ix1-1]
    ul[ix1]=rhou[ix1-1]/rho[ix1-1]
    htotl[ix1]=rhoe[ix1-1]/rho[ix1-1]+p[ix1-1]/rho[ix1-1]
    
    rhor[ix1]=rho[ix1]
    ur[ix1]=rhou[ix1]/rho[ix1]
    htotr[ix1]=rhoe[ix1]/rho[ix1]+p[ix1]/rho[ix1]
  #rhol=rho[nghost-1:-nghost]
  #rhor=rho[nghost:-nghost+1]
  #ul=rhou[nghost-1:-nghost]/rhol
  #ur=rhou[nghost:-nghost+1]/rhor
  #htotl=rhoe[nghost-1:-nghost]/rhol-p[nghost-1:-nghost]/rhol
  #htotr=rhoe[nghost:-nghost+1]/rhor-p[nghost:-nghost+1]/rhor
    # get Roe average
    sqrtrhol=np.sqrt(rhol[ix1])
    sqrtrhor=np.sqrt(rhor[ix1])
    isrholr=sqrtrhol+sqrtrhor
    rhoroe=sqrtrhol*sqrtrhor
    uroe=(sqrtrhol*ul[ix1]+sqrtrhor*ur[ix1])/isrholr
    htotroe=(sqrtrhol*htotl[ix1]+sqrtrhor*htotr[ix1])/isrholr
    csroe=np.sqrt((gam1)*(htotroe-0.5*uroe*uroe))
    csroe2=csroe*csroe
    uroe2=uroe*uroe
    # get eigenvalues eigen vectors in [rhoe, rhou, rho]
    # right eigen vectors or P
    P = np.array([[1.,                 1.,                            1.],
                [uroe-csroe,         uroe,                  uroe+csroe],
                [htotroe-csroe*uroe, 0.5*uroe*uroe, htotroe+csroe*uroe]])
  #left eigenvector or P^-1
    na = 0.5/csroe/csroe
    Pinv=np.array([[na*(gam1*uroe2/2.+uroe*csroe), -na*(gam1*uroe+csroe), na*gam1],
                 [1.-na*gam1*uroe2,              gam1*uroe/csroe2,  -gam1/csroe2],
                 [na*(gam1*uroe2/2.-uroe*csroe), -na*(gam1*uroe-csroe),   na*gam1]])
    # eigenvalues matrix or diagonized Jacobian
    lamb = np.array([[ abs(uroe-csroe), 0,             0              ],
                     [0,                abs(uroe),     0              ],
                     [0,                0,             abs(uroe+csroe)]])
    #cons difference at each side of face
    dU = np.array([rho[ix1]-rho[ix1-1],rhou[ix1]-rhou[ix1-1], rhoe[ix1]-rhoe[ix1-1]])
    #get dU projection on eigenvectors (matrix A)
    A = np.dot(P,lamb)
    A = np.dot(A,Pinv)
    phi[:,ix1] = np.dot(A,dU)
    #print(ix1,x1c[ix1], htotr[ix1],p[ix1],0.5*ur[ix1]*ur[ix1])
    # to be added: check negative in intermediate states
#    print(ix1-nghost,'htot',htotroe,rhoe[ix1-1],rhoe[ix1],p[ix1-1],p[ix1],rho[ix1-1],rho[ix1])
 #   print(ix1-nghost,Pinv)
  #compute Roe flux
  phiroe = 0.5*(flux[:,0:nx1-2*nghost+1]+flux[:,1:nx1+2-2*nghost])-0.5*phi[:,nghost:nx1-nghost+1]
  dF = phiroe[:,1:]-phiroe[:,0:-1]
  rho[nghost:nx1-nghost] = rho[nghost:nx1-nghost] - dt*dF[0,:]/dx1[nghost:-nghost]
  rhou[nghost:nx1-nghost] = rhou[nghost:nx1-nghost] - dt*dF[1,:]/dx1[nghost:-nghost]
  rhoe[nghost:nx1-nghost] = rhoe[nghost:nx1-nghost] - dt*dF[2,:]/dx1[nghost:-nghost]
  print('flux',np.shape(flux))
  print('flux',flux)

  boundaryad( rho, rhou, rhoe, key_bc, nghost)
  
  return dt


def boundaryad(rho, rhou,rhoe, key_bc, nghost):
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



#initial
N=200

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
p=np.zeros(N+2*nghost)
u=np.zeros(N+2*nghost)






#rho=1.+0.3*np.exp(-(x1c-xmid)**2/dg**2)
cs=0.1
for i in range(N+2*nghost):
  if x1f[i]<=50.:
    rho[i]=1.0
    p[i]=1.0
  else:
    rho[i]=0.125
    p[i]=0.1

q=func_prim2cons(rho,u,p,gamma)
rho=q[0,:]
rhou=q[1,:]
rhoe=q[2,:]

#plt.plot(x1c,rhoe/rho-rhou*rhou/rho/rho/2.)
plt.plot(x1c,rho)
plt.plot(x1c,rhou/rho)
plt.plot(x1c,rhoe/rho)
plt.show()

count=0

while count<0:
  if RECONS:
    u=rhou/rho
    eth=rhoe/rho-0.5*u*u
    p=(gamma-1)*rho*eth
    cs=np.sqrt(gamma*p/rho)

    dx1=x1f[1:]-x1f[0:-1]
    nx1=len(x1c)
    dum1 = dx1/(cs+abs(u))
    dt1 = cfl * min(dum1)
    dt = dt1

    #rhof=reconstruct(x1c, x1f,rho,key_m='donorcell')
    #rhouf=reconstruct(x1c, x1f,rhou,key_m='donorcell')
    #rhoef=reconstruct(x1c, x1f,rhoe,key_m='donorcell')
    rhof=reconstruct(x1c, x1f,rho)
    rhouf=reconstruct(x1c, x1f,rhou)
    rhoef=reconstruct(x1c, x1f,rhoe)
    dF = roestepf(x1c[nghost:nx1-nghost], x1f[nghost:nx1-nghost+1], rhof, rhouf, rhoef)
    rho[nghost:nx1-nghost] = rho[nghost:nx1-nghost] - dt*dF[0,:]/dx1[nghost:-nghost]
    rhou[nghost:nx1-nghost] = rhou[nghost:nx1-nghost] - dt*dF[1,:]/dx1[nghost:-nghost]
    rhoe[nghost:nx1-nghost] = rhoe[nghost:nx1-nghost] - dt*dF[2,:]/dx1[nghost:-nghost]
    boundaryad( rho, rhou, rhoe, 'open', nghost)
  else:
    dt=roestep(x1c, x1f, rho, rhou, rhoe, cfl)
  plt.plot(x1c,rho)
  count+=1
  plt.show() 


# set up figure and animation
fig = plt.figure(figsize=(8,6))
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
                      xlim=(0,101),ylim=(0., 4.),ylabel='u')
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
    if RECONS:
      u=rhou/rho
      eth=rhoe/rho-0.5*u*u
      p=(gamma-1)*rho*eth
      cs=np.sqrt(gamma*p/rho)

      dx1=x1f[1:]-x1f[0:-1]
#      nx1=len(x1c)
      dum1 = dx1/(cs+abs(u))
      dt1 = cfl * min(dum1)
      dt = dt1
    
      rhof=reconstruct(x1c, x1f,rho)
      rhouf=reconstruct(x1c, x1f,rhou)
      rhoef=reconstruct(x1c, x1f,rhoe)
      #rhof=reconstruct(x1c, x1f,rho,key_m='donorcell')
      #rhouf=reconstruct(x1c, x1f,rhou,key_m='donorcell')
      #rhoef=reconstruct(x1c, x1f,rhoe,key_m='donorcell')
      qfl = np.array([rhof[0],rhouf[0],rhoef[0]])
      qfr = np.array([rhof[1],rhouf[1],rhoef[1]])
      ##### fixing instability with high order reconstruction  #######
      FfR = func_flux(qfl,gamma);
      FfL = func_flux(qfr,gamma);
      print(np.shape(rhof[0]),np.shape(FfR[0]),np.shape(dx1))
      rhof[0][1:] = rhof[0][1:]+0.5*dt*(FfR[0][:-1]-FfL[0][1:])/dx1[nghost:-nghost]
      rhof[1][1:-1] = rhof[1][1:-1]+0.5*dt*(FfR[0][1:-1]-FfL[0][2:])/dx1[nghost:-nghost-1]
      rhouf[0][1:] = rhouf[0][1:]+0.5*dt*(FfR[1][:-1]-FfL[1][1:])/dx1[nghost:-nghost]
      rhouf[1][1:-1] = rhouf[1][1:-1]+0.5*dt*(FfR[1][1:-1]-FfL[1][2:])/dx1[nghost:-nghost-1]
      rhoef[0][1:] = rhoef[0][1:]+0.5*dt*(FfR[2][:-1]-FfL[2][1:])/dx1[nghost:-nghost]
      rhoef[1][1:-1] = rhoef[1][1:-1]+0.5*dt*(FfR[2][1:-1]-FfL[2][2:])/dx1[nghost:-nghost-1]
      ##### end of fixing instability with high order reconstruction  #######

      #rhof=reconstruct(x1c, x1f,rho)
      #uf=reconstruct(x1c, x1f,u)
      #rhouf=uf*rhof
      #pf=reconstruct(x1c, x1f,p)
      #rhoef=0.5*uf*uf*rhof+pf/(gamma-1)
      
      dF = roestepf(x1c[nghost:-nghost], x1f[nghost:-nghost+1], rhof, rhouf, rhoef)
      rho[nghost:-nghost] = rho[nghost:-nghost] - dt*dF[0,:]/dx1[nghost:-nghost]
      rhou[nghost:-nghost] = rhou[nghost:-nghost] - dt*dF[1,:]/dx1[nghost:-nghost]
      rhoe[nghost:-nghost] = rhoe[nghost:-nghost] - dt*dF[2,:]/dx1[nghost:-nghost]
      boundaryad( rho, rhou, rhoe, 'open', nghost)

    else:
      dt = roestep(x1c, x1f, rho, rhou, rhoe,cfl)  
    #dt = hydrostepad(x1c, x1f, rho, rhou, rhoe,cfl)
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
animate(0)
#t1 = time()
#interval = 1000 * dt - (t1 - t0)

ani = animation.FuncAnimation(fig, animate, frames=300,
                              blit=True, init_func=init)
plt.show()
