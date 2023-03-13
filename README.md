# Project 4

## Training Details:
Hidden Layers: 1  
Nodes in Hidden Layer: 20  
Training Samples = 6000  

## Calculations 

### Python
Grind Rate: 100 (iterations) * 6000 (samples) / 124.5 (time) * 10 (batch size) = 48,192  
Cost: 0.03  
Train Accuracy: 97.2%  
Test Accuracy: 90%

### C  
Grind Rate C: 100 (iterations) * 6000 (samples) / 85.6 (time) * 10 (batch size) = 70,000  
Cost: 0.03  
Train Accuracy: 94.5%  
Test Accuracy: 89.8%  

### CUDA
Grind Rate CUDA: 100 (iterations) * 6000 (samples) / 144 (time) * 40 (batch size) = 166,666  
Cost: 0.03  
Train Accuracy: 94.8%  
Test Accuracy: 90.3%  

Times taken for different thread sizes:  
8: 350s
16: 220s
32: 148s

### Best Performance
Cost: 0.02  
Sample Size: 20k  
Train Accuracy: 96.5  
Test Accuracy: 93.1  

## Sample command line argument for execution
./NN 1 20 100 10

### Explanations:
In Cuda, although we can see speed up as we increase the number of threads, the reason for its performance slower than C and python is because of copying large amound of data from CPU to GPU and vice versa. The result can be significantly better, if the datasize is large and the number of input nodes to be processed is significant