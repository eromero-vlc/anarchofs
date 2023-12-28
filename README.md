# AnarchoFS

AnarchoFS is a virtual filesystem that unionizes the content of several directories at different machines. It is based on FUSE and uses MPI to communicate between processes and transfer content.

The initial motivation of this project is to overcome the IO bottleneck of the shared filesystems in advanced computer facilities.

## Installation

Dependencies:

- C++ compiler
- [libfuse](https://github.com/libfuse/libfuse)
- MPI library, for instance [OpenMPI](https://www.open-mpi.org/)

Execute the default `make` action to compile the binary with `CXX` being the MPI compiler wrapper. For instance:
```
make CXX=mpicxx
```

Otherwise, compile directly the file `anarchofs.cc` with the `libfuse` and the MPI include and dependencies. For instance:
```
mpicxx anarchofs.cc `pkg-config fuse3 --cflags --libs` -o anarchofs
```

## Usage

The more simple way is to 

```
mpirun -np <nprocs> ./anarchofs -s -f -o max_threads=1 -o modules=subdir -o subdir=<basedir> <mountpoint>
```

For instance, the following unionizes the local directory `/tmp` from the machines `hostname{0,1}`:
```
mpirun -H hostname0,hostname1 -np 2 mkdir -p ~/tmp_shared
mpirun -H hostname0,hostname1 -np 2 ./anarchofs -s -f -o max_threads=1 -o modules=subdir -o subdir=/tmp ~/tmp_shared &
```

To unmount the virtual filesystem, kill the `mpirun ... anarchofs` process or invoke `fusermount` (or `fusermount3`) on each machine:
```
mpirun -np <nprocs> fusermount3 -u <mountpoint>
```
