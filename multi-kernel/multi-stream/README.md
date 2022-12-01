#### Memory oversubscritpion with page fault mechanism
ubuntu@test-cuda-mauro:~/cuda_test/um-test-cuda$ ./vector_add_base \\
allocating 4985723288 bytes per array \\ 
no errors, time: 5655.069824 \\
\\
#### Direct memory access with data partitioning between CPU-GPU \\
ubuntu@test-cuda-mauro:~/cuda_test/um-test-cuda$ ./vector_add_advise \\
allocating 4985723288 bytes per array \\
no errors, time: 2381.427734 \\
