#include <cstdio>
#include <ctime>
#include <iostream>
#include <cstdlib>
#include <vector>
#include <complex>
#include <algorithm>
#include <sys/time.h>

#include <cuda_runtime.h>
#include <cusolverDn.h>

#define DEBUG
#define nstream 2

#ifdef DEBUG
#define CUSOLVER_CHECK(err) (HandlecusolverError(err, __FILE__, __LINE__))
#define CUDA_CHECK(err) (HandleError(err, __FILE__, __LINE__))
#define CUBLAS_CHECK(err) (HandleBlasError(err, __FILE__, __LINE__))
#else
#define CUSOLVER_CHECK(err) (err)
#define CUDA_CHECK(err) (err)
#define CUBLAS_CHECK(err) (err)
#endif

static void HandleBlasError(cublasStatus_t err, const char *file, int line)
{

    if (err != CUBLAS_STATUS_SUCCESS)
    {
        fprintf(stderr, "ERROR: %s in %s at line %d (error-code %d)\n",
                cublasGetStatusString(err), file, line, err);
        fflush(stdout);
        exit(-1);
    }
}



static void HandlecusolverError(cusolverStatus_t err, const char *file, int line )
{

    if (err != CUSOLVER_STATUS_SUCCESS)
    {
        fprintf(stderr, "ERROR: %d in %s at line %d, (error-code %d)\n",
                err, file, line, err);
        fflush(stdout);
        exit(-1);
    }
}

static void HandleError(cudaError_t err, const char *file, int line)
{

    if (err != cudaSuccess)
    {
        fprintf(stderr, "ERROR: %s in %s at line %d (error-code %d)\n",
                cudaGetErrorString(err), file, line, err);
        fflush(stdout);
        exit(-1);
    }
}

template <typename T> void print_matrix(const int &m, const int &n, const T *A, const int &lda);

template <> void print_matrix(const int &m, const int &n, const float *A, const int &lda) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            std::printf("%0.2f ", A[j * lda + i]);
        }
        std::printf("\n");
    }
}

template <> void print_matrix(const int &m, const int &n, const double *A, const int &lda) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            std::printf("%0.2f ", A[j * lda + i]);
        }
        std::printf("\n");
    }
}

template <> void print_matrix(const int &m, const int &n, const cuComplex *A, const int &lda) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            std::printf("%0.2f + %0.2fj ", A[j * lda + i].x, A[j * lda + i].y);
        }
        std::printf("\n");
    }
}

template <>
void print_matrix(const int &m, const int &n, const cuDoubleComplex *A, const int &lda) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            std::printf("%0.2f + %0.2fj ", A[j * lda + i].x, A[j * lda + i].y);
        }
        std::printf("\n");
    }
}

int matmul_c_stream(const int m,  cuDoubleComplex *A_, cuDoubleComplex *B_,  const int nmat_ ) {

    // properties of matrix
    const int lda = m;
    int nmat = 20;
    int nnn = 100;
    nmat = nnn;
    cublasOperation_t transa=CUBLAS_OP_N;
    cublasOperation_t transb=CUBLAS_OP_N;

    // cublas setting variebles
    cublasHandle_t cublasH;
    cudaStream_t stream[nstream];

//     printf("solving %d %dx%d matrices multiply with %d streams.\n",nmat,m,m, nstream);
    // eigen storage and workspace
    cuDoubleComplex *A; // matrix should be stored in pinned memory
    cuDoubleComplex *B; // matrix should be stored in pinned memory

    CUDA_CHECK(cudaMallocHost((void **)&A,sizeof(cuDoubleComplex)*lda * m * nmat));
    CUDA_CHECK(cudaMallocHost((void **)&B,sizeof(cuDoubleComplex)*lda * m * nmat));

    // copy to pinned memory
    printf("Copy matrix to pinned memory.\n");
    for (int i=0;i<nmat;i++) {
      std::copy(A_,A_+lda*m,A+i*lda*m);
      std::copy(B_,B_+lda*m,B+i*lda*m);
    }
//     // std::cout << A[m*m-1].x << std::endl;
//     // std::cout << B[m*m-1].x << std::endl;

//     printf("?\n");
    cuDoubleComplex *d_A;
    cuDoubleComplex *d_B;
    cuDoubleComplex *d_C;

    cuDoubleComplex alpha;
    cuDoubleComplex beta;
    alpha = {1.0,0.0};
    beta = {0.0,0.0};
    
//     // step 0: allocate device memory
//     printf("allocate device memory.\n");
    CUBLAS_CHECK(cublasCreate(&cublasH));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_A), sizeof(cuDoubleComplex) * lda * m * nmat));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_B), sizeof(cuDoubleComplex) * lda * m * nmat));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_C), sizeof(cuDoubleComplex) * lda * m * nmat));

    for (int i=0; i < nstream; i++ ) {
      CUDA_CHECK(cudaStreamCreate(&stream[i]));
    //   CUDA_CHECK(cudaStreamCreateWithFlags(&stream[i], cudaStreamNonBlocking));
    }

//     /* step 3: Copy Host To Device */
    CUDA_CHECK(
        cudaMemcpy(d_A, A, sizeof(cuDoubleComplex) * lda * m * nmat, cudaMemcpyHostToDevice ));
    CUDA_CHECK(
        cudaMemcpy(d_B, B, sizeof(cuDoubleComplex) * lda * m * nmat, cudaMemcpyHostToDevice ));

    // CUDA timer
    cudaEvent_t start[nstream], stop[nstream];
    for (int i = 0 ; i < nstream; i++) {
    CUDA_CHECK(cudaEventCreate(&start[i]));
    CUDA_CHECK(cudaEventCreate(&stop[i]));
    }
    for (int i=0; i < nstream; i++ ) {
      CUDA_CHECK(cudaEventRecord(start[i],stream[i]));
   }

    // C timer
    std::clock_t c_start = std::clock();
    struct timeval begin, end;
    gettimeofday(&begin, 0);

    // Main part
    for (int i=0; i < nnn; i++ ) {
    printf("begin inner loop %d in %d \n",i,nnn);
    int ist = i%nstream;
    printf("begin cublasCreate\n");

    CUBLAS_CHECK(cublasSetStream(cublasH, stream[ist]));

    /* step 5: compute matmul  */
    printf("begin cublasZgemm\n");
    CUBLAS_CHECK(cublasZgemm(cublasH, transa, transb, m, m, m, 
        &alpha, &d_A[i*m*lda], lda, &d_B[i*m*lda], lda, &beta, &d_C[i*m*lda], lda));

    }
    CUBLAS_CHECK(cublasDestroy(cublasH));
    for (int i = 0; i< nstream; i++){
        CUDA_CHECK(cudaEventRecord(stop[i],stream[i]));
    }

    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaFreeHost(A));
    CUDA_CHECK(cudaFreeHost(B));
    printf("cudaFree\n");
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    // // // C timer: CPU time
    std::clock_t c_end = std::clock();

    long double time_elapsed_ms = 1000.0 * (c_end-c_start) / CLOCKS_PER_SEC;
    std::cout << "CPU time used in cusolver: " << time_elapsed_ms << " ms\n";

    // C timer: WALL time
    gettimeofday(&end, 0);
    long seconds = end.tv_sec - begin.tv_sec;
    long microseconds = end.tv_usec - begin.tv_usec;
    double elapsed = seconds + microseconds*1e-6;
    
    printf("Wall Time measured: %.3f seconds.\n", elapsed);

    // CUDA timer
    for (int i=0; i< nstream; i++) {
    CUDA_CHECK(cudaEventSynchronize(stop[i]));
    float elapsed_time;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_time, start[i], stop[i]));
    float t_sum = 0;
    t_sum += elapsed_time;

    printf("The %d stream CUDA event time: %gs\n",i,t_sum/1000.0);
    }

    for (int i=0; i< nstream; i++) {
    CUDA_CHECK(cudaEventDestroy(start[i]));
    CUDA_CHECK(cudaEventDestroy(stop[i]));
    }

    // destroy streams
    for (int i=0; i < nstream; i++ ) {
      CUDA_CHECK(cudaStreamDestroy(stream[i]));
    }

    // reset device
    CUDA_CHECK(cudaDeviceReset());

    return EXIT_SUCCESS;
}