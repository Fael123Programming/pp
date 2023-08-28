#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

int* a, * b, * c;

_global_ void vecAdd(int* a, int* b, int* c) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    c[i] = a[i] + b[i];
}

int main() {
    cudaDeviceReset();  // Destrói o contexto do cuda fazendo com que todas as alocações de memória sejam removidas.
    /*  Criação de 3 ponteiros para vetores de inteiros.
        d_a será usado para receber os valores do vetor a dentro do contexto cuda (na GPU).
        d_b será usado para receber os valores do vetor b dentro do contexto cuda (na GPU).
        d_c será usado para receber os valores do vetor c dentro do contexto cuda (na GPU).*/
    int* d_a, * d_b, * d_c;  
    int n = 524288; // Quantidade de posições do vetor para somar.
    int size = n * sizeof(int);  // Quantidade total de memória que deverá ser alocada.

    a = (int*) malloc(size);  // Alocação de memória para vetor a.
    b = (int*) malloc(size);  // Alocação de memória para vetor b.
    c = (int*) malloc(size);  // Alocação de memória para vetor c.

    cudaMalloc((void**) &d_a, size);  // Alocação de memória na GPU para vetor d_a.
    cudaMalloc((void**) &d_b, size);  // Alocação de memória na GPU para vetor d_b.
    cudaMalloc((void**) &d_c, size);  // Alocação de memória na GPU para vetor d_c.

    for (int i = 0; i < n; i++) {  // Preenche os vetores a e b com valores de 0 até (n - 1).
        a[i] = i;
        b[i] = i;
    }

    cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);  // Copia os valores presentes no vetor a para o vetor d_a na GPU.
    cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);  // Copia os valores presentes no vetor b para o vetor d_b na GPU.

    int threads = 256;  // Quantidade de threads por bloco (máximo até 1024).

    int blocks = (n + threads - 1) / threads;  // Quantidade de blocos.

    /*  Chama a função vecAdd (que representa o kernel criado) com a quantidade de blocos e threads para serem usados.
        Passa como argumentos, os vetores alocados em GPU.
    */
    vecAdd << <blocks, threads> >> (d_a, d_b, d_c);  

    /*  Espera até que o dispositivo termine o processamento atual.
        Interrompe até que todas as tarefas requisitadas estejam prontas. 
    */
    cudaDeviceSynchronize();

    cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);  // Copia os valores presentes no vetor d_c na GPU para o vetor c na memória ram.

    printf("\nResult:\n");

    for (int i = 0; i < n; i++)  // Imprime os valores do vetor c no console.
        printf("%d\n", c[i]);

    // Libera toda a memória reservada para os vetores na GPU.
    cudaFree(d_a);  
    cudaFree(d_b);
    cudaFree(d_c);

    // Libera toda a memória reservada para os vetores na ram.
    free(a);
    free(b);
    free(c);

    return 0;
}