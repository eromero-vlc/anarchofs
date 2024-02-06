CXXFLAGS ?= -Wall -Wextra -O0 -g -std=c++11 -DANARCOFS_LOG
CXXFLAGS ?= -Wall -O3 $(shell pkg-config fuse3 --cflags ) -DAFS_DAEMON_USE_FUSE
#CXXFLAGS ?= -Wall -Wextra -g3 -O0 -DANARCOFS_LOG $(shell pkg-config fuse3 --cflags ) # for debug
LIBS ?= $(shell pkg-config fuse3 --libs )

anarchofs: anarchofs.cc anarchofs_lib.h Makefile
	${CXX} ${CXXFLAGS} anarchofs.cc ${LIBS} -o anarchofs

test_socket: test_socket.cc anarchofs_lib.h Makefile
	${CXX} ${CXXFLAGS} test_socket.cc -o test_socket

clean:
	rm -f anarchofs

format:
	clang-format -i anarchofs.cc anarchofs_lib.h test_socket.cc

prepare_fuse_test_dirs: unmount_test
	rm -rf t0 t1 tref
	mkdir t0
	seq 1 10 > t0/f0
	seq 1 10 > t0/c
	mkdir t1
	seq 101 110 > t1/f1
	seq 11 20 > t1/c
	mkdir tref
	seq 1 10 > tref/f0
	seq 1 20 > tref/c
	seq 101 110 > tref/f1
	mkdir -p v0
	mkdir -p v1

unmount_test:
	-pkill -9 anarchofs &> /dev/null
	-mpirun -q -np 1 fusermount3 -u v0 : -np 1 fusermount3 -u v1 &> /dev/null

TEST_OPTIONS ?= -s -f -o max_threads=1
#TEST_OPTIONS ?= -s -d -o max_threads=1

run_fuse_test: prepare_fuse_test_dirs
	mpirun -np 1 ./anarchofs ${TEST_OPTIONS} -o modules=subdir -o subdir=${PWD}/t@NPROC ./v0 : -np 1 ./anarchofs ${TEST_OPTIONS} -o modules=subdir -o subdir=${PWD}/t@NPROC ./v1 & \
	sleep 1; \
	for d in v0 v1; do \
		for f in c f0 f1; do \
			cmp -s $$d/$$f tref/$$f || echo "failed tref/$$f for process $$d"; \
		done; \
	done || true
	mpirun -np 1 fusermount3 -u v0 : -np 1 fusermount3 -u v1

run_socket_test:
	rm -rf v0 v1
	mkdir v0 v1
	nohup mpirun -np 1 -x AFS_SOCKET=/tmp/afs0.sock ./anarchofs : -np 1 -x AFS_SOCKET=/tmp/afs1.sock ./anarchofs &> afs.out &
	mpirun -np 1 -x AFS_SOCKET=/tmp/afs0.sock ./test_socket v@NPROC : -np 1 -x AFS_SOCKET=/tmp/afs1.sock ./test_socket v@NPROC
	killall anarchofs
