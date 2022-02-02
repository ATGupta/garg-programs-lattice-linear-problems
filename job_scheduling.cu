#include <stdio.h>

__global__ void run_job_scheduling(volatile int*, volatile int*, volatile int*, volatile int*, int, volatile int*, volatile int*, volatile int*);
__device__ void monitor(volatile int*, volatile int*, int);
__device__ void read_global_state(volatile int*, int, volatile int*, int);
__device__ bool B(volatile int*, volatile int*, int, int);
__device__ bool forbidden(volatile int*, volatile int*, int, int);
__device__ int alpha(volatile int*, int, int, volatile int*, volatile int*);
__device__ int get_readG_value(volatile int*, int, int, int);
__device__ int get_time_taken(volatile int*, int j, int);
__device__ int getDependency(volatile int* input, int, int, int);

// un-comment the print statement(s) in run_job_scheduling() to get a view of what is happening
int main() {
	int process_count = 5;
	int input[] = {11,11,11,12,14,16,2,3,1,2,2,0,1,2,1,3};
	/*
	INPUT FORMAT (the following 4 lines are with respect to only the above input)
	position i: 0...process_count-1 = start position of prerequisites of i
	position process_count = length of array
	position process_count+1...2*process_count + 1 = time taken by each process
	position 2*process_count+1...3*process_count+1 = prerequisites of processes
	
	some functions in this program are made just so that the program is consistent with the input format.
	
	the input can be generally modified by imagining the format as follows: SEQUENTIALLY:
	[process_count](start position of prerequisites of i in input),
	[1](length of array),
	[process_count](time taken by each process),
	[length of array - start position of prerequisites of process 0](prerequisites of processes).
	*/
	
	int* G = (int*)malloc(process_count * sizeof(int));
	int T[] = {10, 10, 10, 10, 10};
	int* readG = (int*)malloc(process_count * process_count * sizeof(int));
	int* execution_status = (int*)malloc(process_count * sizeof(int));
	for(int i=0; i < process_count; i++) {
		G[i] = 0;
		execution_status[i] = 0;
	}
	
	int* halt_bit = (int*)malloc(sizeof(int)); halt_bit[0] = 0;
	
	int* input_changed = (int*)malloc(process_count * sizeof(int));
	for(int i=0; i < process_count; i++) input_changed[i] = 1;
	
	int *input_cuda, *G_cuda, *T_cuda, *readG_cuda, *halt_bit_cuda, *input_changed_cuda, *execution_status_cuda;
	
	cudaMalloc(&input_cuda, input[process_count] * sizeof(int));
	cudaMalloc(&G_cuda, process_count * sizeof(int));
	cudaMalloc(&T_cuda, process_count * sizeof(int));
	cudaMalloc(&readG_cuda, process_count * process_count * sizeof(int));
	cudaMalloc(&halt_bit_cuda, sizeof(int));
	cudaMalloc(&input_changed_cuda, process_count * sizeof(int));
	cudaMalloc(&execution_status_cuda, process_count * sizeof(int));
	
	cudaMemcpy(input_cuda, input, input[process_count] * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(G_cuda, G, process_count * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(T_cuda, T, process_count * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(readG_cuda, readG, process_count * process_count * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(halt_bit_cuda, halt_bit, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(input_changed_cuda, input_changed, process_count * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(execution_status_cuda, execution_status, process_count * sizeof(int), cudaMemcpyHostToDevice);
	
	run_job_scheduling<<<(process_count + 255)/256, 256>>>(G_cuda, readG_cuda, T_cuda, input_cuda, process_count, input_changed_cuda, halt_bit_cuda, execution_status_cuda);
	
	cudaMemcpy(G, G_cuda, process_count * sizeof(int), cudaMemcpyDeviceToHost);
		
	printf("The final vector is (start time of all processes):\n");
	for(int i = 0; i < process_count; i++) {
		printf("%d\t", G[i]);
	}
	printf("\n");
}

__device__ int IDLE = 0, FALSE = 0;
__device__ int BUSY = 1, TRUE = 1;

__global__
void run_job_scheduling(volatile int* G, volatile int* readG, volatile int* T, volatile int* input, int process_count, volatile int* input_changed, volatile int* halt_bit, volatile int* execution_status) {
	int j = blockIdx.x*blockDim.x + threadIdx.x;
	if(j >= process_count + 1) return;
	
	for(;;j++) {
	//this loop is made just in case if the GPU decides to provide us with less threads than desired.
		
		if(halt_bit[0] == 1) return;
		
		if(execution_status[j] == IDLE) {
			execution_status[j] = BUSY;
			if(j == process_count) {
				monitor(input_changed, halt_bit, process_count);
				//printf("%d\t%d\t%d\t%d\t%d\n",G[0],G[1],G[2],G[3],G[4]);
			}
			else {
				read_global_state(G, j, readG, process_count);
				//printf("thread%d.\t%d\t%d\t%d\t%d\t%d\t...\t%d\n",j,readG[5*j+0],readG[5*j+1],readG[5*j+2],readG[5*j+3],readG[5*j+4],B(readG,input,j,process_count));
				if(forbidden(readG, input, j, process_count)) {
					int ALPHA = alpha(readG, j, process_count, T, input);
					if(ALPHA == -1) {
						input_changed[j] = FALSE;
						G[j] = -1;
						return;
					}
					G[j] = ALPHA;
					input_changed[j] = TRUE;
				}
				else
					input_changed[j] = FALSE;
			}
			execution_status[j] = IDLE;
		}
		if(j >= process_count) 
		// if the for loop is converted to an infinite loop, then this if condition will also be not needed
			j = 0;
	}
}

__device__
void monitor(volatile int* input_changed, volatile int* halt_bit, int process_count) {
	bool all_set = true;
	for(int i = 0; i < process_count; i++) {
		if(input_changed[i] == TRUE)
			all_set = false;
	}
	if(all_set)
		halt_bit[0] = 1;
}

__device__
bool B(volatile int* readG, volatile int* input, int j, int process_count) {
	for(int iLoop = input[j]; iLoop < input[j+1]; iLoop++) {
		int i_in_pre_j = input[iLoop];
		if(get_readG_value(readG, j, j, process_count) < get_readG_value(readG, j, i_in_pre_j, process_count) + get_time_taken(input, j, process_count))
			return false;
	}
	return true;
}

__device__
bool forbidden(volatile int* readG, volatile int* input, int j, int process_count) { 
        if(B(readG, input, j, process_count))
                return false;
        else
                return true;
}

__device__
void read_global_state(volatile int* G, int j, volatile int* readG, int process_count) {
	for(int i = 0; i < process_count; i++) {
		readG[j * process_count + i] = G[i];
	}
}

__device__
int alpha(volatile int* readG, int j, int process_count, volatile int* T, volatile int* input) {
	int val_g = get_readG_value(readG, j, j, process_count) + 1;
	
	for(; val_g <= T[j]; val_g++) {
		int t = readG[j * process_count + j];
		readG[j * process_count + j] = val_g;
		
		if(B(readG, input, j, process_count)) return val_g;
		
		readG[j * process_count + j] = t;
	}
	return -1;
}

__device__
int get_readG_value(volatile int* readG, int j, int entry, int process_count) {
	return readG[j * process_count + entry];
}

__device__ 
int get_time_taken(volatile int* input, int j, int process_count) {
	return input[process_count + j + 1];
}

__device__
int getDependency(volatile int* input, int j, int entry, int process_count) {
	return input[input[j] + entry];
}
