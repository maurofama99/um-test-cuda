Test set used for understanding CUDA Unified Memory, in particular the interaction between static and managed memory allocation.

### single_kernel
One kernel running, different tests use different type of memory allocation
 
### multi_kernel
Two kernels running, testing with only managed memory allocation and mixed (static and managed) memory allocation
### multi_stream
Same tests using CUDA streams, prefetch and latest CUDA features such as MemAdvise
