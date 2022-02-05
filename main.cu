int main() {
	1. input = INPUT
	2. initialize G, T, readG.
	3. halt_bit=0.
	4. input_changed = all ones.
	5. run<<<blocks, number of cores per block>>>(G, readG, T, input, input size, input_changed, halt-bit);
	6. fetch output. read G.
	7. print results
}

__global__
void run(input arguments) {
	1. i = get thread ID.
	2. if i>input size, then return.
	3. if i == 0, then
	4. 	monitor(G)
	5. else
	6. 	readG = read_global_state(G, i)
	7. 	if forbidden(readG, i)
	8. 		ALPHA = alpha(readG, j)
	9. 		if ALPHA == -1, then
	10. 			input_changed[i] = 0
	11. 			return -1
	12. 		G[i] = ALPHA
	13. 		input_changed[i] = 1;
	14. 	else
	15. 		input_changed[i] = 0
}

__device__
bool B(G, i) {
	if i satisfies the predicate in G, then
		return true.
	else
		return false
}

__device__
bool B(G) {
	if all i satisfy the predicate in G, then
		return true
	else
		return false 
}

__device__
bool forbidden(readG, i) {
	if B(G, i), then
		return false
	else
		return true
}
