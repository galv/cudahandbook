all:
	nvcc -g -DDEBUG --gpu-architecture sm_50 -I../chLib/ --compiler-bindir /usr/bin/g++-4.9 -o atomic_speed_test atomic_speed_test.cu

clean:
	rm atomic_speed_test
