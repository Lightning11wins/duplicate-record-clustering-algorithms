#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdint.h>
#include <string.h>

#include "lib/utils.h"

// Run with the following command (didn't feel like making a make file.)
// clear;clear;gcc sparse_dot.c -lm -Wall -g;./a.out

void handle_token(double token, size_t* remaining, double* value) {
	if (token < 0.0) {
		// This run contains -token zeros.
		*remaining = (size_t)llround(-token);
		*value = 0.0;
	} else {
		// This run contains one value.
		*remaining = 1;
		*value = token;
	}
}

/* Compute dot product of two encoded sparse vectors.
Encoding rules:
- positive double x => one coordinate with value x
- negative double -k => k zero coordinates
*/
double dot_sparse(const double *v1, const double *v2, int num_dims) {
	double val1 = 0.0, val2 = 0.0, result = 0.0;
	size_t vec1_remaining = 0, vec2_remaining = 0;
	size_t loc = 0, i1 = 0, i2 = 0;
	while (loc < num_dims) {
		if (vec1_remaining == 0) handle_token(v1[i1++], &vec1_remaining, &val1);
		if (vec2_remaining == 0) handle_token(v2[i2++], &vec2_remaining, &val2);

		// Accumulate the result. If either vector is 0 here,
		// the total is 0 and this statement does nothing.
		result += val1 * val2;
		
		// Consume overlap from both runs.
		size_t overlap = min(vec1_remaining, vec2_remaining);
		vec1_remaining -= overlap;
		vec2_remaining -= overlap;
		loc += overlap;
	}
	return result;
}

double* convert_vector(double* vector, size_t len) {
	double* out = (double*)malloc(len * sizeof(double));
	for (int i = 0, loc = 0; loc < len;) {
		double token = vector[i++];
		if (token > 0.0) out[loc++] = token;
		else repeat(-token, j) out[loc++] = 0.0;
	}
	return out;
}

double dot_prod(double* v1, double* v2, int len) {
	double dot_product = 0;
	for (int i = 0; i < len; i++) {
		dot_product += v1[i] * v2[i];
	}
	return dot_product;
}

double* gen_vector(int length) {
	int pos = 0, loc = 0;
	double vector[length];
	while (loc < length) {
		if (rand() % 2) {
			int max_zeros = min(length - loc, 8);
			int zeros = random(1, max_zeros);
			vector[pos++] = -((double)zeros);
			loc += zeros;
		} else {
			double val = (double)random(1, 2048);
			vector[pos++] = val;
			loc += 1;
		}
	}
	
	double* heap_vector = (double*) malloc(pos * sizeof(double) + 1);
    if (!heap_vector) fprintf(stderr, "malloc(%d) - FAIL\n", pos);
    memcpy(heap_vector, vector, pos * sizeof(double));
	heap_vector[pos++] = NAN;
	return heap_vector;
}

int main(void) {
	printf("start\n");
	/* Example:
	 * v1: 1, 2, -3       => 1,   2, 0, 0, 0
	 * v2: 0.5, -1, 4, -2 => 0.5, 0, 4, 0, 0
	 * Expected dot = 0.5
	 */
	srand(1842637);
	for (int itr = 0; itr < 2048; itr++) {
		size_t len = 8u;
		double* v1_sparse = gen_vector(len);
		double* v2_sparse = gen_vector(len);
		printf("v1s: %g", v1_sparse[0]);
		for (int i = 1; !isnan(v1_sparse[i]); i++) {
			printf(", %g", v1_sparse[i]);
		}
		printf("\n");
		
		printf("v2s: %g", v2_sparse[0]);
		for (int i = 1; !isnan(v2_sparse[i]); i++) {
			printf(", %g", v2_sparse[i]);
		}
		printf("\n");
		
		double* v1 = convert_vector(v1_sparse, len);
		double* v2 = convert_vector(v2_sparse, len);
		printf("v1: %g", v1[0]); repeat(len - 1, i) printf(", %g", v1[i + 1]); printf("\n");
		printf("v2: %g", v2[0]); repeat(len - 1, i) printf(", %g", v2[i + 1]); printf("\n");

		double dot_s = dot_sparse(v1_sparse, v2_sparse, len);
		printf("dos = %.10g\n", dot_s);

		double dot = dot_prod(v1, v2, len);
		printf("dot = %.10g\n", dot);
		
		free(v1_sparse);
		free(v2_sparse);
		free(v1);
		free(v2);
		
		if (dot_s - dot > 0.01) {
			fprintf(stderr, "Oh no!\n");
			return 1;
		}
	}
	return 0;
}
