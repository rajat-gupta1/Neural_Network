python:
	python3 NN.py 1 20 100 10
C:
	gcc NN3.c -o NN -lm
Cuda:
	nvcc -arch sm_70 NN.cu -o NN