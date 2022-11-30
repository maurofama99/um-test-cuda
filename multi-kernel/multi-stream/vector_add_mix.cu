#include <cstdio>
#include <cstdlib>

#define gpuErrchk(ans) { gpuAssert( (ans), __FILE__, __LINE__ ); }

inline void
gpuAssert( cudaError_t code, const char * file, int line, bool abort = true )
{
        if ( cudaSuccess != code )
        {
                fprintf( stderr, "\nGPUassert: %s %s %d\n", cudaGetErrorString( code ), file, line );
                if ( abort )
                        exit( code );
        }

return;

} /* gpuAssert */

__global__ void Add( int N ,int Offset ,float * devA , float * devB , float *devC )
{

        for ( int idx = blockIdx.x * blockDim.x + threadIdx.x + Offset; idx < N; idx += blockDim.x * gridDim.x )

                devC[ idx ] = devA[ idx ] + devB[ idx ];

}

int main() {

        size_t free_t, total_t;

        gpuErrchk( cudaMemGetInfo(&free_t, &total_t) );
        
        //int N = 4000000;
        //unsigned long N = 1395864368;  //c.a. 1.3 * 4 GB, tot 15.6 GB
        //unsigned long N = 1288490184;  //c.a. 1.2 * 4 GB, tot 14.4 GB
        unsigned long N = 0.31 * free_t;
        
        printf("allocating %lu bytes per array\n", N);
        
        int Threads = 1024;

        const int NbStreams = 8;
        
        /************************************
                  HOST ALLOCATION
        ************************************/
        float *A , *B , *C1, *C2;
        gpuErrchk( cudaHostAlloc( (void**) &A , N * sizeof(*A) ,cudaHostAllocDefault ) );
        gpuErrchk( cudaHostAlloc( (void**) &B , N * sizeof(*B) ,cudaHostAllocDefault ) );
        gpuErrchk( cudaHostAlloc( (void**) &C1 , N * sizeof(*C1) ,cudaHostAllocDefault ) );
        gpuErrchk( cudaHostAlloc( (void**) &C2 , N * sizeof(*C2) ,cudaHostAllocDefault ) );
        
        for ( int i = 0; i < N; i++ ) {
                A[ i ] = 1;
                B[ i ] = 2;
                C1[i] = 0;
                C2[i] = 0;
        }
        
        /************************************
                  STREAM CREATION
        ************************************/
        cudaStream_t Stream1[ NbStreams ];
        for ( int i = 0; i < NbStreams; i++ )
        	gpuErrchk( cudaStreamCreate(&Stream1[ i ]) );
        
        cudaStream_t Stream2[ NbStreams ];
        for ( int i = 0; i < NbStreams; i++ )
        	gpuErrchk( cudaStreamCreate(&Stream2[ i ]) );

        const int StreamSize = N / NbStreams;
        
        /************************************
               MANAGED DEVICE ALLOCATION
        ************************************/
        float *d_A, *d_B, *d_C;
        gpuErrchk( cudaMallocManaged(  &d_A , N * sizeof(*d_A)) );
        gpuErrchk( cudaMallocManaged(  &d_B , N * sizeof(*d_B)) );
        gpuErrchk( cudaMallocManaged(  &d_C , N * sizeof(*d_C)) );
        
        gpuErrchk( cudaMemPrefetchAsync(  d_B , N * sizeof(*d_B), 0) );
        gpuErrchk( cudaMemPrefetchAsync(  d_A , N * sizeof(*d_A), 0) );
        gpuErrchk( cudaMemPrefetchAsync(  d_C , N * sizeof(*d_C), 0) );
        
        /************************************
                 EXECUTION ON MANAGED
        ************************************/
        for ( int i = 0; i < NbStreams; i++ )
        {
                int Offset = i * StreamSize;

                gpuErrchk( cudaMemcpyAsync(&d_A[ Offset ], &A[ Offset ], StreamSize * sizeof(*A), cudaMemcpyHostToDevice, Stream2[ i ]) );
                gpuErrchk( cudaMemcpyAsync(&d_B[ Offset ], &B[ Offset ], StreamSize * sizeof(*B), cudaMemcpyHostToDevice, Stream2[ i ]) );

                Add<<< StreamSize / Threads, Threads, 0, Stream2[ i ]>>>( Offset+StreamSize ,Offset, d_A , d_B , d_C );

                gpuErrchk( cudaMemcpyAsync(&C2[ Offset ], &d_C[ Offset ], StreamSize * sizeof(*d_C), cudaMemcpyDeviceToHost, Stream2[ i ]) );

        }
        
        /************************************
               STATIC DEVICE ALLOCATION
        ************************************/
        float *devA , *devB , *devC;
        gpuErrchk( cudaMalloc( (void**) &devA , N * sizeof(*devA)) );
        gpuErrchk( cudaMalloc( (void**) &devB , N * sizeof(*devB)) );
        gpuErrchk( cudaMalloc( (void**) &devC , N * sizeof(*devC)) );
  
        /************************************
                     EXECUTION
        ************************************/
        for ( int i = 0; i < NbStreams; i++ )
        {
                int Offset = i * StreamSize;

                gpuErrchk( cudaMemcpyAsync(&devA[ Offset ], &A[ Offset ], StreamSize * sizeof(*A), cudaMemcpyHostToDevice, Stream1[ i ]) );
                gpuErrchk( cudaMemcpyAsync(&devB[ Offset ], &B[ Offset ], StreamSize * sizeof(*B), cudaMemcpyHostToDevice, Stream1[ i ]) );

                Add<<< StreamSize / Threads, Threads, 0, Stream1[i]>>>( Offset+StreamSize ,Offset, devA , devB , devC );

                gpuErrchk( cudaMemcpyAsync(&C1[ Offset ], &devC[ Offset ], StreamSize * sizeof(*devC), cudaMemcpyDeviceToHost, Stream1[ i ]) );

        }
        
        cudaDeviceSynchronize();
        
        /************************************
                    RESULT CHECK
        ************************************/
        for ( int i = 0; i < N; i++ ) {
                if (C1[i] != (A[i]+B[i])) {
                printf("mismatch at %d, was: %f, should be: %f (first)\n", i, C1[i], (A[i]+B[i])); return 1;
        	}
        }
        
                for ( int i = 0; i < N; i++ ) {
                if (C2[i] != (A[i]+B[i])) {
                printf("mismatch at %d, was: %f, should be: %f (second)\n", i, C2[i], (A[i]+B[i])); return 1;
        	}
        }

        // DESTROY CONTEXT
        for ( int i = 0; i < NbStreams; i++ )
                gpuErrchk( cudaStreamDestroy(Stream1[ i ]) );
                for ( int i = 0; i < NbStreams; i++ )
                gpuErrchk( cudaStreamDestroy(Stream2[ i ]) );

        gpuErrchk( cudaFree(devA) );
        gpuErrchk( cudaFree(devB) );
        gpuErrchk( cudaFree(devC) );

        gpuErrchk( cudaFreeHost(A) );
        gpuErrchk( cudaFreeHost(B) );
        gpuErrchk( cudaFreeHost(C1) );

        return 0;

}
