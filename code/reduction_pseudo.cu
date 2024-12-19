#include <stdio.h>
#include <sys/time.h>

#ifndef tileSize
    #define tileSize 16
#else
    #if tileSize < 1 || tileSize > 32
        #error "tileSize debe ser un valor entre 1 e 32"
    #endif
#endif

#ifndef iters
    #define iters 1000
#endif

#define INIT_TIME(prev, init) \
    gettimeofday(&prev, NULL); \
    gettimeofday(&init, NULL);

// remove overhead created by call to gettimeofday
#define GET_TIME(prev, init, final, res) \
    gettimeofday(&final, NULL); \
    res = (final.tv_sec-init.tv_sec+(final.tv_usec-init.tv_usec)/1.e6) - \
          (init.tv_sec-prev.tv_sec+(init.tv_usec-prev.tv_usec)/1.e6);

/*
 Calcula a suma de val no grupo g. O array x, temporal e en memoria distribuída,
 ten que ser o suficientemente grande para conter g.size() enteiros. O resultado
 esperado será (n-1)*n/2, tendo en conta que o primeiro fío ten rango 0
*/
__device__ int sumReduction(int *x, int val, int groupDim, int lane) {
    // redución, de tal xeito que o resultado quede en val no fío con rango 0
    for (int i = groupDim / 2; i > 0; i /= 2) {
        x[lane] = val;
        __syncthreads();
        if (lane < i) { val += x[lane + i]; }
        __syncthreads();
    }

    // o fío con rango 0 devolve o resultado, o resto -1
    if (lane == 0) { return val; }
    else { return -1; }
}

// Kernel, crea grupos cooperativos e realiza reducións
__global__ void cgkernel() {
    // array temporal para a redución
    extern __shared__ int workspace[];

    int input, output, expectedOutput;
    input = threadIdx.x;

    // resultado esperado, usando a fórmula previamente mencionada
    expectedOutput = (blockDim.x - 1) * blockDim.x / 2;
    
    output = sumReduction(workspace, input, blockDim.x, threadIdx.x);

    // o fío mestre imprime o resultado
    if (threadIdx.x == 0) {
        printf("Suma de 0 a %d no grupo é %d, esperado %d\n",
            blockDim.x - 1, output, expectedOutput);
        printf("Agora creando %d \"subgrupos\" de %d fíos\n",
                blockDim.x / tileSize, tileSize);
    }

    __syncthreads();

    // subgrupos unidimensionais de tileSize fíos
    int threadGroup;
    for (int i = 0; i < blockDim.x / tileSize; i++) {
        if (threadIdx.x < tileSize * (i + 1)) {
            threadGroup = i;
            break;
        }
    }

    // offset para que cada subgrupo teña unha parte distinta do array
    int workspaceOffset = threadGroup * tileSize;

    input = threadIdx.x % tileSize;

    expectedOutput = (tileSize - 1) * tileSize / 2;

    output = sumReduction(workspace + workspaceOffset, input, tileSize, input);

    // o fío mestre de cada subgrupo imprime o resultado
    if (input == 0) {
        printf("Suma de 0 a %d no subgrupo %d é %d, esperado %d\n",
            tileSize - 1, threadGroup, output, expectedOutput);
    }

    return;
}

int main() {
    struct timeval init, prev, final;
    double time;

    INIT_TIME(prev, init);
    for (int _ = 0; _ < iters; _++) {
        int blocksPerGrid = 1;
        int threadsPerBlock = 1024;

        printf("Usando %d bloques de %d fíos\n", blocksPerGrid, threadsPerBlock);

        cgkernel<<<blocksPerGrid, threadsPerBlock, threadsPerBlock * sizeof(int)>>>();
        cudaError_t err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            fprintf(stderr, "Erro de CUDA: %s\n", cudaGetErrorString(err));
            return 1;
        }
    }
    GET_TIME(prev, init, final, time);
    printf("Tempo total: %f s\n", time);

    FILE *fp = fopen("pseudo.csv", "a");
    fprintf(fp, "%d,%d,%f\n", tileSize, iters, time);
    fclose(fp);

    return 0;
}
