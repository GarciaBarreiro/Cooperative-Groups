/* Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include <stdio.h>
#include <cooperative_groups.h>
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

using namespace cooperative_groups;

/*
 Calcula a suma de val no grupo g. O array x, temporal e en memoria distribuída,
 ten que ser o suficientemente grande para conter g.size() enteiros. O resultado
 esperado será (n-1)*n/2, tendo en conta que o primeiro fío ten rango 0
*/
__device__ int sumReduction(thread_group g, int *x, int val) {
    // rango do fío dentro do grupo
    int lane = g.thread_rank();

    // redución, de tal xeito que o resultado quede en val no fío con rango 0
    for (int i = g.size() / 2; i > 0; i /= 2) {
        x[lane] = val;
        g.sync();
        if (lane < i) { val += x[lane + i]; }
        g.sync();
    }

    // o fío con rango 0 devolve o resultado, o resto -1
    if (lane == 0) { return val; }
    else { return -1; }
}

// Kernel, crea grupos cooperativos e realiza reducións
__global__ void cgkernel() {
    // grupo con tódolos fíos do bloque
    thread_block threadBlockGroup = this_thread_block();
    int threadBlockGroupSize = threadBlockGroup.size();

    // array temporal para a redución
    extern __shared__ int workspace[];

    int input, output, expectedOutput;
    input = threadBlockGroup.thread_rank();

    // resultado esperado, usando a fórmula previamente mencionada
    expectedOutput = (threadBlockGroupSize - 1) * threadBlockGroupSize / 2;
    
    output = sumReduction(threadBlockGroup, workspace, input);

    // o fío mestre imprime o resultado
    if (threadBlockGroup.thread_rank() == 0) {
        printf("Suma de 0 a %d no grupo é %d, esperado %d\n",
            threadBlockGroupSize - 1, output, expectedOutput);
        printf("Agora creando %d subgrupos de %d fíos\n",
            threadBlockGroupSize / tileSize, tileSize);
    }

    // mesmo resultado neste caso que usar __syncthreads()
    threadBlockGroup.sync();

    // subgrupos unidimensionais de tileSize fíos
    thread_block_tile<tileSize> tiledPartition = tiled_partition<tileSize>(threadBlockGroup);

    // offset para que cada subgrupo teña unha parte distinta do array
    int workspaceOffset = threadBlockGroup.thread_rank() - tiledPartition.thread_rank();

    input = tiledPartition.thread_rank();

    expectedOutput = (tileSize - 1) * tileSize / 2;

    output = sumReduction(tiledPartition, workspace + workspaceOffset, input);

    // o fío mestre de cada subgrupo imprime o resultado
    if (tiledPartition.thread_rank() == 0) {
        printf("Suma de 0 a %d no subgrupo %d é %d, esperado %d\n",
            tileSize, tiledPartition.meta_group_rank(), output, expectedOutput);
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

    FILE *fp = fopen("groups.csv", "a");
    fprintf(fp, "%d,%d,%f\n", tileSize, iters, time);
    fclose(fp);

    return 0;
}
