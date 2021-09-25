import numpy as np
from mpi4py import MPI

comm = MPI.COMM_WORLD

rank = comm.Get_rank()
size = comm.Get_size()

data = (rank+1)**2
print(data,rank)
data = comm.gather(data, root=0)
print(data,rank)

print("AAAAAA")
data=np.linspace(rank,rank+0.4,6)
comm.Barrier()
print(data,rank,"old")
if rank <size-1:
#  data=np.arange(100, dtype=np.float64)
  comm.Send(data[4:6], dest=rank+1)
elif rank > 0:
#  data = np.empty(100, dtype=np.float64)
  comm.Recv(data[0:2], source=rank-1)
comm.Barrier()
if rank == 0:
  print("BBBBB\n")

print(data,rank)

#if rank == 0:
#  for i in range(size):
#    assert data[i] == (i+1)**2
#else:
#  assert data is None

#if rank == 0:
#  print(data)
