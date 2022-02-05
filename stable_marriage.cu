#include <stdio.h>

__global__ void run_stable_marriage(volatile int*, volatile int*, volatile int*, volatile int*, volatile int*, volatile int*, volatile int*);
__device__ void monitor(volatile int*, volatile int*, int);
__device__ void read_global_state(volatile int*, int, volatile int*, volatile int*);
__device__ bool B(volatile int*, volatile int*, int);
__device__ bool forbidden(volatile int*, volatile int*, int);
__device__ int alpha(volatile int*, int, volatile int*, volatile int*);
__device__ int get_readG_value(volatile int*, volatile int*, int, int);
__device__ int mpref(volatile int*, volatile int*, int, int);
__device__ int rank(volatile int*, int, int);

// un-comment the print statement(s) in run_job_scheduling() to get a view of what is happening
int main() {
	int men = 3, women = 3;
	int input[] = {men,women,2,0,1,2,0,1,0,1,2,1,2,0,2,1,0,0,1,2};
	/*
	INPUT FORMAT (the following 4 lines are with respect to only the above input)
	position 0 = #men
	position 1 = #women
	position 2 ... 1 + men * women = menwize proposal preferences for women
	position 2 + men * women ... 1 + 2 * men * women = womenwise rank for men
	
	some functions in this program are made just so that the program is consistent with the input format.
	
	the input can be generally modified by imagining the format as follows: SEQUENTIALLY:
	[1](number of men),
	[1](number of women),
	[men * women](menwize proposal preferences for women),
	[women * men](womenwise rank for men).
	*/
	
	int process_count = men;
	int* G = (int*)malloc(process_count * sizeof(int));
	int T[] = {women, women, women};
	int* readG = (int*)malloc(men * women * sizeof(int));
	int* execution_status = (int*)malloc(process_count * sizeof(int));
	for(int i=0; i < process_count; i++) {
		G[i] = 0;
		execution_status[i] = 0;
	}
	
	int* halt_bit = (int*)malloc(sizeof(int)); halt_bit[0] = 0;
	
	int* input_changed = (int*)malloc(process_count * sizeof(int));
	for(int i=0; i < process_count; i++) input_changed[i] = 1;
	
	int *input_cuda, *G_cuda, *T_cuda, *readG_cuda, *halt_bit_cuda, *input_changed_cuda, *execution_status_cuda;
	
	cudaMalloc(&input_cuda, (2*men*women+2) * sizeof(int));
	cudaMalloc(&G_cuda, process_count * sizeof(int));
	cudaMalloc(&T_cuda, process_count * sizeof(int));
	cudaMalloc(&readG_cuda, men * women * sizeof(int));
	cudaMalloc(&halt_bit_cuda, sizeof(int));
	cudaMalloc(&input_changed_cuda, process_count * sizeof(int));
	cudaMalloc(&execution_status_cuda, process_count * sizeof(int));
	
	cudaMemcpy(input_cuda, input, (2*men*women+2) * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(G_cuda, G, process_count * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(T_cuda, T, process_count * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(readG_cuda, readG, men * women * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(halt_bit_cuda, halt_bit, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(input_changed_cuda, input_changed, process_count * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(execution_status_cuda, execution_status, process_count * sizeof(int), cudaMemcpyHostToDevice);
	
	run_stable_marriage<<<(process_count + 255)/256, 256>>>(G_cuda, readG_cuda, T_cuda, input_cuda, input_changed_cuda, halt_bit_cuda, execution_status_cuda);
	
	cudaMemcpy(G, G_cuda, process_count * sizeof(int), cudaMemcpyDeviceToHost);
		
	printf("The final vector (regret of men; first choice is regret zero):\n");
	for(int i = 0; i < process_count; i++) {
		printf("%d\t", G[i]);
	}
	printf("\n");
}

__device__ int IDLE = 0, FALSE = 0;
__device__ int BUSY = 1, TRUE = 1;

__global__
void run_stable_marriage(volatile int* G, volatile int* readG, volatile int* T, volatile int* input, volatile int* input_changed, volatile int* halt_bit, volatile int* execution_status) {
	int process_count = input[0];
	
	int j = blockIdx.x*blockDim.x + threadIdx.x;
	if(j >= process_count + 1) return;
	
	for(;;j++) {
	//this loop is made just in case if the GPU decides to provide us with less threads than desired.
		
		if(halt_bit[0] == 1) return;
		
		if(execution_status[j] == IDLE) {
			execution_status[j] = BUSY;
			if(j == process_count) {
				monitor(input_changed, halt_bit, process_count);
				//printf("%d\t%d\t%d\n",G[0],G[1],G[2]);
			}
			else {
				read_global_state(G, j, readG, input);
				//printf("thread%d.\t%d\t%d\t%dt...\t%d\n",j,readG[3*j+0],readG[3*j+1],readG[3*j+2],B(readG,input,j));
				if(forbidden(readG, input, j)) {
					int ALPHA = alpha(readG, j, T, input);
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
bool B(volatile int* readG, volatile int* input, int man) {
	int men = input[0], women = input[1];
	int z = mpref(input, readG, man, man);
	
	for(int i = 0; i < men; i++) {
		if(i != man){
			int z_prime = mpref(input, readG, man, i);
			if(z == z_prime) {
				if(rank(input, z, man) > rank(input, z_prime, i)) {
					return false;
				}
			}
		}
	}
	return true;
}

__device__
bool forbidden(volatile int* readG, volatile int* input, int man) { 
        if(B(readG, input, man))
                return false;
        else
                return true;
}

__device__
void read_global_state(volatile int* G, int man, volatile int* readG, volatile int* input) {
	int men = input[0], women = input[1];
	for(int i = 0; i < women; i++) {
		readG[man * women + i] = G[i];
	}
}

__device__
int alpha(volatile int* readG, int man, volatile int* T, volatile int* input) {
	int men = input[0], women = input[1];
	int val_g = get_readG_value(readG, input, man, man) + 1;
	
	for(; val_g <= T[man]; val_g++) {
		int t = readG[man * women + man];
		readG[man * women + man] = val_g;
		
		if(B(readG, input, man)) return val_g;
		
		readG[man * women + man] = t;
	}
	return -1;
}

__device__
int get_readG_value(volatile int* readG, volatile int* input, int man, int entry) {
	int men = input[0], women = input[1];
	return readG[man * women + entry];
}

__device__ 
int mpref(volatile int* input, volatile int* readG, int from_the_perspective_of, int man) {
	int men = input[0], women = input[1];
	int choice_number = get_readG_value(readG, input, from_the_perspective_of, man);
	return input[2 + man * women + choice_number];
}

__device__
int rank(volatile int* input, int woman, int man) {
	int men = input[0], women = input[1];
	int rank = 0, start = 2 + men * women + woman * men;
	for(int i = 0; i < men; i++) {
		if(input[start + i] == man) {
			rank = i;
			break;
		}
	}
	return rank;
}
