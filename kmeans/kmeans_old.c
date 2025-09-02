#include <ctype.h>
#include <execinfo.h>
#include <float.h>
#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <signal.h>

#include "lib/arraylist.h"
#include "lib/timer.h"
#include "lib/utils.h"

#define INDENT "\t> "
#define NUM_DIMS 251
#define SEED 1621963727
#define DUPE_THRESHOLD 0.75
#define KMEANS_IMPROVEMENT_THRESHOLD 0.0002

// Test Parameters
unsigned int window_sizes[] = {3, 6, 16, 32, 64, 256};
unsigned int cluster_counts[] = {2, 4, 8, 16, 32, 64, 128, 256, 512, 1024};
unsigned int dataset_sizes[] = {100, 1000, 5000, 10000, 50000, 100000, 1000000, 3975429};
unsigned int max_iter = 64;

FILE* complete_file = NULL;
FILE* sliding_file = NULL;
FILE* kmeans_file = NULL;

/*** Complexity Analysis
 *** n: The number of input records.
 *** d: The number of dimensions. (251)
 *** w: The number of records in sliding window.
 *** k: The number of kmeans clusters.
 *** i: The number of kmeans iterations.
 *** s: The number of records in the average cluster.
 ***/
 
// ====================================
// Helper functions
 
// 1M: 185.66 MB
void fprint_mem(FILE* out) {
	FILE *fp = fopen("/proc/self/statm", "r");
	if (!fp) { perror("fopen"); return; }

	long size, resident, share, text, lib, data, dt;
	if (fscanf(fp, "%ld %ld %ld %ld %ld %ld %ld",
			&size, &resident, &share, &text, &lib, &data, &dt) != 7) {
		fprintf(stderr, "Failed to read memory info\n");
		fclose(fp);
		return;
	}
	fclose(fp);

	long page_size = sysconf(_SC_PAGESIZE); // in bytes
	long resident_bytes = resident * page_size;

	fprintf(out, INDENT"Memory used (RSS): %ld bytes (%.2f MB)\n", resident_bytes, (double)resident_bytes / (1024.0 * 1024.0));
	fprintf(out, INDENT"Share %ldb, Text %ldb, Lib %ldb, Data %ldb\n", share, text, lib, data);
}

void fprint_vector(FILE* out, const int* vector) {
	for (unsigned int j = 0, idx = 0; idx < NUM_DIMS; j++) {
		if (vector[j] > 0) { fprintf(out, " % 3d,", vector[j]); idx++; }
		else {
			unsigned int num = (unsigned)(-vector[j]);
			idx += num;
			repeat (num, k) fprintf(out, "    ,");
		}
	}
}

void fprint_centroid(FILE* out, const double* vector) {
	repeat (NUM_DIMS, i) fprintf(out, " % 3g,", vector[i]);
}

// Stacktrace that lies to you. Idk why.
void fprint_stacktrace(void) {
	// Define an array to hold the return addresses.
	void *callstack[64];
	int frames = backtrace(callstack, 64);
	char** symbol_list = backtrace_symbols(callstack, frames);
	if (symbol_list == NULL) { return perror("backtrace_symbols failed"); }

	// Print the backtrace.
	fprintf(stderr, "Stacktrace (%d):\n", frames);
	repeat (frames, i) {
		char* symbol = symbol_list[i];
		
		// Attempt to get print line number.
		unsigned int line_number = 0;
		if (!strncmp(symbol, "./bin/kmeans_debug.(", 20u)) goto fail;
		char* start = strchr(symbol, '+');
		if (start == NULL) goto fail;
		if (sscanf(++start, "%x", &line_number) != 1) goto fail;
		char fname[BUFSIZ];
		snprintf(memset(fname, 0, sizeof(fname)), max(sizeof(fname), (unsigned)((unsigned long)start - (unsigned long)symbol - 20u)), "%s", symbol + 20);
		fprintf(stderr, "./kmeans.c#%s:%u\n", fname, line_number);
		continue;
		
		fail:
		fprintf(stderr, "%s\n", symbol);
	}
	free(symbol_list);
}

// Function for failing on error.
void fail(const char* function_name) {
	// Create the most descriptive error message we can.
	char error_buf[BUFSIZ];
	snprintf(error_buf, sizeof(error_buf), "kmeans.c: Fail - %s", function_name);
	perror(error_buf);
	
	// Throw segfault for easier debugging.
	fprintf(stderr, "Program will now segfault.\n");
	raise(SIGSEGV);
}

/*** Helper function for compact error handling on library & system function calls.
 *** Any non-zero value is treated as an error, exiting the program.
 ***
 *** @param result The result of the function we're checking.
 *** @param function_name The name of the function being checked (for debugging).
 *** @returns result
 ***/
int check(int result, const char* function_name) {
	if (result != 0) fail(function_name);
	return result;
}

/*** Helper function for compact error handling on library & system function calls.
 *** Any null value is treated as an error, exiting the program.
 ***
 *** @param result The result of the function we're checking.
 *** @param function_name The name of the function being checked (for debugging).
 *** @returns result
 ***/
void* check_ptr(void* result, const char* function_name) {
	if (result == NULL) fail(function_name);
	return result;
}

/*** Safe malloc with error handling.
 *** 
 *** @param size The size of memory to malloc.
 *** @returns Clean, allocated memory.
 ***/
void* mallocs(const size_t size) {
	void* ptr = malloc(size);
	if (ptr == NULL) {
		// Create the most descriptive error message we can.
		char error_buf[BUFSIZ];
		snprintf(error_buf, sizeof(error_buf), "kmeans.c: Fail - malloc(%lu bytes)", size);
		perror(error_buf);
		
		// Throw segfault for easier debugging.
		fprintf(stderr, "Program will now segfault.\n");
		raise(SIGSEGV);
	}
	return memset(ptr, 0, size);
}

// Keep an eye on this function.
unsigned int get_char_pair_hash(unsigned int num1, unsigned int num2) {
	double sum = (num1 * num1 * num1) + (num2 * num2 * num2);
	double scale = ((double)num1 + 1.0) / ((double)num2 + 1.0);
	unsigned int hash = (unsigned int)round(sum * scale) - 1u;
	return hash % NUM_DIMS;
}

/*** Builds a vector using a string.
 *** 
 *** @param strs The string used to build the vector.
 *** @returns The built vector.
 ***/
int* build_vector(char* str) {
	// Malloc space for a dense vector.
	unsigned int* dense_vector = (unsigned int*)mallocs(NUM_DIMS * sizeof(unsigned int));
	
	// j is the former character, i is the latter.
	int num_chars = (int)strlen(str);
	for (int j = -1, i = 0; i <= (int)num_chars; i++) {
		// If latter character is punctuation or whitespace, skip it.
		if (ispunct(str[i]) || isspace(str[i])) continue;
		
		unsigned int temp1 = (j == -1) ? '`' : (unsigned int)tolower(str[j]);
		unsigned int temp2 = (i == num_chars) ? '`' : (unsigned int)tolower(str[i]);

		// If either character is a number, reassign the code point
		// The significance of 75 here seems to be that it puts the numbers
		// right after the end of the lowercase letters, but they will still
		// colide with the {, |, }, ~ and DEL ASCII characters.
		if ('0' <= temp1 && temp1 <= '9') temp1 += 75u;
		if ('0' <= temp2 && temp2 <= '9') temp2 += 75u;
		
		// Hash the character pair into an index (dimension).
		unsigned int dim = get_char_pair_hash(temp1, temp2);
		
		// Increment the dimension of the dense vector by a number from 1 to 13.
		dense_vector[dim] += (temp1 + temp2) % 13 + 1;
		
		j = i;
	}
	
	// Count how much space is needed for a sparse vector.
	bool zero_prev = false;
	unsigned int size = 0u;
	repeat (NUM_DIMS, j) {
		if (dense_vector[j] == 0u) {
			size += (zero_prev) ? 0u : 1u;
			zero_prev = true;
		} else {
			size++;
			zero_prev = false;
		}
	}

	// Convert the dense vector above to a sparse vector.
	int* sparse_vector = (int*)mallocs(size * sizeof(int));
	unsigned int j = 0u, sparse_idx = 0u;
	while (j < NUM_DIMS) {
		if (dense_vector[j] == 0u) {
			// Count and store consecutive zeros, except the first one,
			// which we already know is zero.
			unsigned int zero_count = 1;
			j++;
			while (j < NUM_DIMS && dense_vector[j] == 0u) {
				zero_count++;
				j++;
			}
			sparse_vector[sparse_idx++] = (int)-zero_count;
		} else {
			// Store the value
			sparse_vector[sparse_idx++] = (int)dense_vector[j++];
		}
	}
	
	// Free unused dense vector.
	free(dense_vector);
	
	return sparse_vector;
}

/*** Compute the value for `k` (number of clusters), given a dataset of with
 *** a size of `n`.
 *** 
 *** The following table shows data sizes vs.selected cluster size. In testing,
 *** these numbers tended to givea good balance of accuracy and dulocates detected.
 *** 
 *** ```csv
 *** Data Size, Actual
 *** 10k,       12
 *** 100k,      33
 *** 1M,        67
 *** ```
 *** 
 *** This function is not intended for datasets smaller than (`n < ~2000`).
 *** These should be handled using complete search.
 *** 
 *** LaTeX Notation: \log_{36}\left(n\right)^{3.1}-8
 *** 
 *** @param n The size of the dataset.
 *** @returns k, the number of clusters to use.
 ***/
unsigned int compute_k(unsigned int n) {
	return (unsigned)max(2, pow(log(n) / log(36), 3.2) - 8);
}

double magnitude_sparse(const int* vector) {
	unsigned int magnitude = 0;
	for (unsigned int i = 0u, dim = 0u; dim < NUM_DIMS;) {
		const int val = vector[i++];
		
		// Negative val represents -val 0s in the array, so skip that many values. 
		if (val < 0) dim += (unsigned)(-val);
		
		// We have a value, so square it and add it to the magnitude.
		else {magnitude += (unsigned)(val * val); dim++;}
	}
	return sqrt((double)magnitude);
}

double magnitude_dense(const double* vector) {
	double magnitude = 0.0;
	repeat (NUM_DIMS, i) magnitude += vector[i] * vector[i];
	return sqrt(magnitude);
}

void parse_token(const int token, unsigned int* remaining, unsigned int* value) {
	if (token < 0) {
		// This run contains -token zeros.
		*remaining = (unsigned)(-token);
		*value = 0u;
	} else {
		// This run contains one value.
		*remaining = 1;
		*value = (unsigned)(token);
	}
}

/*** Calculate similarity on sparce vectors.
 *** 
 *** @param v1 Sparse vector 1.
 *** @param v1 Sparse vector 2.
 *** @returns Similarity between 0 and 1 where
 *** 	1 indicates completely different and
 *** 	0 indicates identical.
 *** 
 *** Complexity: `O(3d)`
 ***/
double sparse_similarity(const int* v1, const int* v2) {
	// Calculate dot product.
	unsigned int val1 = 0, val2 = 0, dot_product = 0;
	unsigned int vec1_remaining = 0, vec2_remaining = 0;
	unsigned int dim = 0, i1 = 0, i2 = 0;
	while (dim < NUM_DIMS) {
		if (vec1_remaining == 0) parse_token(v1[i1++], &vec1_remaining, &val1);
		if (vec2_remaining == 0) parse_token(v2[i2++], &vec2_remaining, &val2);

		// Accumulate the dot_product. If either vector is 0 here,
		// the total is 0 and this statement does nothing.
		dot_product += val1 * val2;
		
		// Consume overlap from both runs.
		unsigned int overlap = min(vec1_remaining, vec2_remaining);
		vec1_remaining -= overlap;
		vec2_remaining -= overlap;
		dim += overlap;
	}
	
	// Optional optimization to speed up nonsimilar vectors.
	if (dot_product == 0) return 0.0;
	
	// Return the difference score.
	double total_magnitude = magnitude_sparse(v1) * magnitude_sparse(v2);
	double similarity = (double)dot_product / total_magnitude;
	if (similarity < 0.0 || 1.000000001 < similarity) {
		FILE* out = stdout;
		fprintf(out, "Strange similarity: %g\n", similarity);
		fprintf(out, "v1:\t"); fprint_vector(out, v1); fprintf(out, "\n");
		fprintf(out, "v2:\t"); fprint_vector(out, v2); fprintf(out, "\n");
		fprintf(out, "\n");
		fflush(out);
	}
	return similarity;
}
// #define sparse_dif(v1, v2) (1.0 - sparse_similarity(v1, v2))

double sparse_similarity_c(const int* v1, const double* c2) {
	// Calculate dot product
	double dot_product = 0.0;
	for (unsigned int i = 0u, dim = 0u; dim < NUM_DIMS;) {
		const int val = v1[i++];
		
		// Negative val represents -val 0s in the array, so skip that many values. 
		if (val < 0) dim += (unsigned)(-val);
		
		// We have a value, so square it and add it to the magnitude.
		else dot_product += (double)val * c2[dim++];
	}
	
	// Return the difference score.
	double total_magnitude = magnitude_sparse(v1) * magnitude_dense(c2);
	double similarity = dot_product / total_magnitude;
	if (similarity < 0.0 || 1.000000001 < similarity) {
		FILE* out = stdout;
		fprintf(out, "Strange similarity_c: %g\n", similarity);
		fprintf(out, "v1:\t"); fprint_vector(out, v1); fprintf(out, "\n");
		fprintf(out, "c2:\t"); fprint_centroid(out, c2); fprintf(out, "\n");
		fprintf(out, "\n");
		fflush(out);
	}
	return similarity;
}
#define sparse_dif_c(v1, c2) (1.0 - sparse_similarity_c(v1, c2))

/*** Calculate the average size of all clusters in a set of vectors.
 ***
 *** @param vectors The vectors of the dataset.
 *** @param num_vectors The number of vectors in the dataset.
 *** @param labels The clusters to which vectors are assigned.
 *** @param centroids The locations of the centroids.
 *** @returns The average cluster size.
 ***/
double print_cluster_size(int** vectors, unsigned int num_vectors, unsigned int* labels, double** centroids, unsigned int num_clusters, unsigned int iteration) {
	double cluster_sums[num_clusters];
	double noncluster_sums[num_clusters];
	unsigned int cluster_sizes[num_clusters];
	memset(cluster_sums, 0, sizeof(cluster_sums));
	memset(noncluster_sums, 0, sizeof(noncluster_sums));
	memset(cluster_sizes, 0, sizeof(cluster_sizes));
	
	// Sum the difference from each vector to its cluster centroid.
	repeat (num_vectors, i) {
		unsigned int label = labels[i];
		int* vector = vectors[i];
		cluster_sums[label] += sparse_dif_c(vector, centroids[label]);
		cluster_sizes[label]++;
		
		for (unsigned int j = 0; j < num_clusters; j++) {
			if (j == label) continue;
			noncluster_sums[j] += sparse_dif_c(vector, centroids[j]);
		}
	}
	
	// Calculate the average difference per cluster and then the overall average.
	// fprintf(kmeans_file, "Cluster Sizes:\n");
	double cluster_total = 0.0, noncluster_total = 0.0;
	double max_cluster_size = 0.0, min_cluster_size = 1.0;
	unsigned int max_cluster_label = 0, min_cluster_label = 0, num_valid_clusters = 0;
	repeat (num_clusters, label) {
		unsigned int cluster_count = cluster_sizes[label];
		unsigned int noncluster_count = num_vectors - cluster_count;
		if (cluster_count == 0) continue;
		
		double cluster_size = cluster_sums[label] / cluster_count;
		double noncluster_size = noncluster_sums[label] / noncluster_count;
		cluster_total += cluster_size;
		noncluster_total += noncluster_size;
		num_valid_clusters++;
		
		if (cluster_size > max_cluster_size) {
			max_cluster_size = cluster_size;
			max_cluster_label = label;
		}
		if (cluster_size < min_cluster_size) {
			min_cluster_size = cluster_size;
			min_cluster_label = label;
		}
		
		// fprintf(kmeans_file,
		// 	"> Cluster #%d (x%d): %.4lf (vs. %.4lf).\n",
		// 	label, cluster_size, cluster_size, noncluster_size
		// ); // Debug
	}
	
	// Verify that there are valid clusters.
	if (num_valid_clusters == 0) {
		printf("kmeans #%u: No valid clusters!\n", iteration);
		return 0.0;
	}
	
	// Calculate average sizes.
	double average_cluster_size = cluster_total / num_valid_clusters;
	double average_noncluster_size = noncluster_total / num_valid_clusters;
	
	// Print data
	fprintf(kmeans_file,
		"\nkmeans #%u:\n"
			INDENT"Average cluster: %.4lf\n"
			INDENT"Average noncluster: %.4lf\n"
			INDENT"Largest cluster: #%d @ %.4lf\n"
			INDENT"Smallest cluster: #%d @ %.4lf\n",
		iteration,
		average_cluster_size,
		average_noncluster_size,
		max_cluster_label, max_cluster_size,
		min_cluster_label, min_cluster_size
	);
	
	return average_cluster_size;
}

void select_centroids(int** vectors, unsigned int num_vectors, double** centroids, unsigned int num_clusters) {
	// Setup.
	srand(SEED);
	double* weight = (double*)mallocs(num_vectors * sizeof(double));
	
	// Prevent picking the same vector twice.
	bool* is_chosen = (bool*)mallocs(num_vectors * sizeof(bool));
	repeat (num_vectors, i) is_chosen[i] = false;

	// Pick first centroid uniformly at random.
	unsigned int first_index = (unsigned int)rand() % num_vectors;
	is_chosen[first_index] = true;
	{
		double* centroid = centroids[0];
		memset(centroid, 0, NUM_DIMS * sizeof(double));
		const int *vector = vectors[first_index];
		for (unsigned int i = 0u, dim = 0u; dim < NUM_DIMS;) {
			int token = vector[i++];
			if (token < 0) { dim += (unsigned)(-token); }
			else { centroid[dim++] = (double)token; }
		}
	}

	// Initialize weight to the first centroid.
	repeat (num_vectors, i) {
		if (is_chosen[i]) { weight[i] = 0.0; continue; }
		double dist = sparse_dif_c(vectors[i], centroids[0]);
		// Use squared distance as weight (kmeans++ typical choice).
		weight[i] = dist * dist;
	}

	// Choose remaining centroids.
	for (unsigned int c = 1; c < num_clusters; c++) {
		double total_weight = 0.0;
		repeat (num_vectors, i) total_weight += weight[i];

		unsigned int next_idx = 0;
		if (total_weight <= 0.0) {
			fprintf(stderr, "Too many clusters: k=%u, datasize=%u, total_weight=%.4lf\n",num_clusters, num_vectors, total_weight);
			exit(42);
		}
		
		// Select with probability proportional to weight (weights).
		double r = ((double)rand() / (double)RAND_MAX) * total_weight;
		double acc = 0.0;
		repeat (num_vectors, i) {
			acc += weight[i];
			if (r <= acc) { next_idx = i; break; }
		}

		// Mark is_chosen and copy sparse vector into dense centroid.
		is_chosen[next_idx] = true;
		double* centroid = centroids[c];
		memset(centroid, 0, NUM_DIMS * sizeof(double));
		const int* vector = vectors[next_idx];
		for (unsigned int i = 0u, dim = 0u; dim < NUM_DIMS;) {
			const int token = vector[i++];
			if (token < 0) { dim += (unsigned)(-token); }
			else { centroid[dim++] = (double)token; }
		}

		// Update weight: for each point, distance = min(old, dist(point, new_centroid))^2
		repeat (num_vectors, i) {
			if (is_chosen[i]) { weight[i] = 0.0; continue; }
			const double dist = sparse_dif_c(vectors[i], centroid);
			weight[i] = min(weight[i], dist * dist);
		}
	}

	free(weight);
	free(is_chosen);
}

/*** Executes the k-means clustering algorithm. Selects NUM_CLUSTERS random
 *** vectors as initial centroids. Then points are assigned to the nearest
 *** centroid, after which centroids are moved to the center of their points.
 ***
 *** @param vectors The vectors to cluster.
 *** @param num_vectors The number of vectors to cluster.
 *** @param labels Stores the final cluster identities of the vectors after
 *** 	clustering is completed.
 *** @param centroids Stores the locations of the centroids used for the clusters
 *** 	of the data.
 *** @param iterations The number of iterations that actually executed is stored
 *** 	here. Leave this NULL if you don't care.
 *** @param max_iter The max number of iterations.
 *** @param num_clusters The number of clusters to generate.
 ***
 *** @attention - Assumes: NUM_CLUSTERS is in scope and is the number of centroids.
 *** @attention - Assumes: MAX_ITER is in scope and is the max number of iterations.
 ***
 *** @attention - Issue: At larger numbers of clustering iterations, some
 *** 	clusters have a size of	negative infinity. In this implementation,
 *** 	the bug is mitigated by setting a small number of max iterations,
 *** 	such as 16 instead of 100.
 *** @attention - Issue: Clusters do not apear to improve much after the first
 *** 	iteration, which puts the efficacy of the algorithm into question. This
 *** 	may be due to the uneven density of a typical dataset. However, the
 *** 	clusters still offer useful information.
 *** 
 *** Complexity:
 *** 
 *** - `O(kd + k + i*(k + n*(k+d) + kd))`
 *** 
 *** - `O(kd + k + ik + ink + ind + ikd)`
 *** 
 *** - `O(nk + nd)`
 ***/
void kmeans(int** vectors, unsigned int num_vectors, unsigned int* labels, double** centroids, unsigned int* iterations, unsigned int max_iter, unsigned int num_clusters, double* debug_time) {
	// Select random vectors to use as the initial centroids.
	srand(SEED);
	repeat (num_clusters, i) {
		// Pick a random vector.
		const unsigned int random_index = (unsigned int)rand() % num_vectors;
		
		// Sparse copy the vector into a densely allocated centroid.
		double* centroid = centroids[i];
		const int* vector = vectors[random_index];
		for (unsigned int i = 0u, dim = 0u; dim < NUM_DIMS;) {
			int token = vector[i++];
			if (token > 0) centroid[dim++] = (double)token;
			else repeat (-token, j) centroid[dim++] = 0.0;
		}
	}
	
	// Allocate memory for new centroids.
	double** new_centroids = (double**)mallocs(num_clusters * sizeof(double*));
	repeat (num_clusters, i) {
		new_centroids[i] = (double*)mallocs(NUM_DIMS * sizeof(double));
	}
	
	// Main kmeans loop.
	double previous_average_cluster_size = 1.0;
	unsigned int* cluster_counts = (unsigned int*)mallocs(num_clusters * sizeof(unsigned int));
	repeat (max_iter, iter) {
		bool changed = false;
		
		// Reset new centroids.
		repeat (num_clusters, i) {
			repeat (NUM_DIMS, dim) {
				new_centroids[i][dim] = 0.0;
			}
			cluster_counts[i] = 0u;
		}
		
		// Assign each point to the nearest centroid.
		repeat (num_vectors, i) {
			const int* vector = vectors[i];
			double min_dist = DBL_MAX;
			unsigned int best_centroid_label = 0;
			
			// Find nearest centroid.
			repeat (num_clusters, j) {
				double dist = sparse_dif_c(vector, centroids[j]);
				if (dist < min_dist) {
					min_dist = dist;
					best_centroid_label = j;
				}
			}
			
			// Update label to new centroid, if necessary.
			if (labels[i] != best_centroid_label) {
				labels[i] = best_centroid_label;
				changed = true;
			}
			
			// Accumulate values for new centroid calculation.
			double* best_centroid = new_centroids[best_centroid_label];
			for (unsigned int i = 0u, dim = 0u; dim < NUM_DIMS;) {
				const int val = vector[i++];
				if (val < 0) dim += (unsigned)(-val);
				else best_centroid[dim++] += (double)val;
			}
			cluster_counts[best_centroid_label]++;
		}
		
		// Save number of iterations (if needed).
		if (iterations != NULL) *iterations = iter + 1;
		
		// Stop if centroids didn't change.
		if (!changed) break;
		
		// Update centroids.
		repeat (num_clusters, i) {
			if (cluster_counts[i] == 0u) continue;
			double* centroid = centroids[i];
			const double* new_centroid = new_centroids[i];
			const unsigned int cluster_count = cluster_counts[i];
			repeat (NUM_DIMS, dim) {
				centroid[dim] = new_centroid[dim] / cluster_count;
			}
		}
		
		// Print cluster size for debugging.
		Timer* timer = timer_new();
		timer_benchmark(timer,
			const double average_cluster_size = print_cluster_size(vectors, num_vectors, labels, centroids, num_clusters, iter);
		);
		*debug_time += timer_get(timer);
		timer_free(timer);
		fflush(stdout);
		
		// Is there enough improvement?
		const double improvement = previous_average_cluster_size - average_cluster_size;
		fprintf(kmeans_file, INDENT"Improvement: %.4lf\n", improvement);
		fflush(kmeans_file);
		if (improvement < KMEANS_IMPROVEMENT_THRESHOLD) break;
		previous_average_cluster_size = average_cluster_size;
	}
	
	// kmeans memory usage.
	fprint_mem(stdout);
	
	// Free memory.
	repeat (num_clusters, i) {
		free(new_centroids[i]);
	}
	free(new_centroids);
	free(cluster_counts);
}

/*** Loads the dataset from disk into the global `dataset` array.
 *** 
 *** @attention - Not thread safe: Uses strdup and strtok.
 *** 
 *** @param dataset_path Path to a comma-separated file containing the dataset strings.
 ***/
void load_dataset(const char* dataset_path, char** dataset, const unsigned int dataset_size) {
	FILE *file = fopen(dataset_path, "r");
	if (!file) {
		char error_buf[BUFSIZ];
		snprintf(error_buf, sizeof(error_buf), "Failed to open file: %s", dataset_path);
		perror(error_buf);
		exit(EXIT_FAILURE);
	}
	
	char* buffer = NULL;
	size_t bufsize = 0;
	int error = getdelim(&buffer, &bufsize, EOF, file) == -1;
	fclose(file);
	
	if (error) {
		char error_buf[BUFSIZ];
		snprintf(error_buf, sizeof(error_buf), "Failed to read file: %s", dataset_path);
		perror(error_buf);
		exit(EXIT_FAILURE);
	}
	
	// Verify dataset size.
	size_t size = 1;
	for (char* buf = buffer; *buf != '\0';) {
        if (*buf == ',') size++;
        buf++;
    }
	if (size != dataset_size) {
		fprintf(stderr, "\nWarning: Expected dataset of size %u but got %ld.\n\n", dataset_size, size);
	}
	
	size_t count = 0;
	char* token = strtok(buffer, ",");
	while (token && count < dataset_size) {
		dataset[count++] = strdup(token);
		token = strtok(NULL, ",");
	}
	
	free(buffer);
}

/*** Scans the entire dataset for duplicates by pairwise similarity.
 *** 
 *** This is the "complete" strategy: every pair `(i, j)` is compared and
 *** pairs with `(similarity > DUPE_THRESHOLD)` are appended to the returned list.
 *** The function also logs found duplicates to the `complete_file`.
 *** 
 *** @param vectors Array of precomputed frequency vectors for all dataset strings.
 *** @param num_vectors The number of vectors to be scanned.
 *** @returns A locked ArrayList containing duplicate index pairs.
 *** 
 *** Complexity: `O(n^2 * 3d)`
 ***/
ArrayList* find_complete_dups(int** vectors, unsigned int num_vectors) {
	ArrayList* complete_dups = al_newc(num_vectors * 2u);
	repeat (num_vectors, i) {
		const int* v1 = vectors[i];
		for (unsigned int j = i + 1; j < num_vectors; j++) {
			const int* v2 = vectors[j];
			if (sparse_similarity(v1, v2) > DUPE_THRESHOLD) {
				al_add(complete_dups, i);
				al_add(complete_dups, j);
			}
		}	
	}
	
	// Lock results.
	al_lock(complete_dups);
	al_trim_to_size(complete_dups);
	
	// Log duplocates found by the complete strategy.
	// fprintf(complete_file, "Duplocates found: x%ld/%u\n", complete_dups->size / 2, num_vectors);
	// for (size_t i = 0; i < complete_dups->size;) {
	// 	unsigned int d1 = al_get(complete_dups, i++);
	// 	unsigned int d2 = al_get(complete_dups, i++);
	// 	fprintf(complete_file, "%s (#%u) & %s (#%u)\n", dataset[d1], d1, dataset[d2], d2);
	// }
	
	return complete_dups;
}

/*** Finds duplicates using a sliding-window strategy.
 *** 
 *** Compares each vector only with the next n pairs in a "sliding window"
 *** that moves down the data. Results are logged to `sliding_file` and
 *** validated against the `complete_dups` list.
 ***
 *** @param vectors Array of precomputed frequency vectors for all dataset strings.
 *** @param num_vectors The number of vectors to be scanned.
 *** @param window_size The size of the sliding window.
 *** @returns A locked ArrayList containing duplicate index pairs.
 *** 
 *** Complexity: `O(nw * 3d)`
 ***/
ArrayList* find_sliding_dups(int** vectors, unsigned int num_vectors, const unsigned int window_size) {
	ArrayList* sliding_dups = al_newc(512);
	repeat (num_vectors, i) {
		const int* v1 = vectors[i];
		const unsigned int j_max = min(i + window_size, num_vectors);
		for (unsigned int j = i + 1; j < j_max; j++) {
			const int* v2 = vectors[j];
			if (sparse_similarity(v1, v2) > DUPE_THRESHOLD) {
				al_add(sliding_dups, i);
				al_add(sliding_dups, j);
			}
		}
	}
	
	// Lock results.
	al_lock(sliding_dups);
	al_trim_to_size(sliding_dups);
	
	// Log and verify duplocates found by the sliding strategy.
	// size_t num_sliding_dups = sliding_dups->size;
	// fprintf(sliding_file, "Duplocates found: x%ld/%d\n", num_sliding_dups, num_vectors);
	// for (size_t i = 0; i < num_sliding_dups;) {
	// 	unsigned int d1 = al_get(sliding_dups, i++);
	// 	unsigned int d2 = al_get(sliding_dups, i++);
	// 	fprintf(sliding_file, "%s (#%d) & %s (#%d)\n", dataset[d1], d1, dataset[d2], d2);
		
	// 	if (!verify_dupe(complete_dups, d1, d2)) {
	// 		printf("sliding found false dupe: %s (#%d) & %s (#%d)\n", dataset[d1], d1, dataset[d2], d2);
	// 	}
	// }
	// fprintf(sliding_file, "\n\n");
	
	return sliding_dups;
}

/*** Runs k-means clustering then finds duplicates inside clusters.
 ***
 *** Steps:
 *** 
 ***  - Allocate labels and centroids.
 *** 
 ***  - Run kmeans(...) to assign labels.
 *** 
 ***  - For each cluster, compare points inside the cluster and record pairs
 ***    with `(similarity > DUPE_THRESHOLD)`.
 *** 
 ***  - Log duplicates and cluster contents to `kmeans_file`, validate found
 ***    pairs against `complete_dups` to detect mistakes.
 ***
 *** @param vectors Array of precomputed frequency vectors for all dataset strings.
 *** @param num_vectors The number of vectors to be scanned.
 *** @param iterations The number of iterations that actually executed is stored
 *** 	here. Leave this NULL if you don't care.
 *** @param max_iter Maximum iterations passed to kmeans().
 *** @param num_clusters Number of clusters to produce.
 *** @returns A locked ArrayList containing duplicate index pairs.
 *** 
 *** Complexity: `O((nk + nd) + 3d*s^2)`
 *** Complexity: `O((nk + nd) + 3d*(n/k)^2)`
 *** Complexity: `O(nk + (n/k)^2)` (ignoring i & d)
 ***/
ArrayList* find_kmeans_dups(int** vectors, unsigned int num_vectors, unsigned int* iterations, unsigned int max_iter, unsigned int num_clusters) {
	// Create timers.
	Timer* timer_total = timer_new();
	Timer* timer_clustering = timer_new();
	Timer* timer_scan = timer_new();
	timer_start(timer_total);
	
	// Malloc memory for finding clusters.
	unsigned int* labels = (unsigned int*)mallocs(num_vectors * sizeof(unsigned int));
	repeat (num_vectors, i) labels[i] = 0;
	double** centroids = (double**)mallocs(num_clusters * sizeof(double*));
	repeat (num_clusters, i) centroids[i] = (double*)mallocs(NUM_DIMS * sizeof(double));
	fprintf(kmeans_file, "\nMalloced...\n");
	fflush(kmeans_file);
	
	// Execute kmeans clustering.
	double debug_time = 0.0;
	timer_benchmark(timer_clustering,
		kmeans(vectors, num_vectors, labels, centroids, iterations, max_iter, num_clusters, &debug_time);
	);
	fprintf(kmeans_file, "Clustered...\n");
	fflush(kmeans_file);
	
	// Find duplocates in clusters.
	timer_benchmark(timer_scan,
		ArrayList* kmeans_dups = al_newc(512);
		repeat (num_vectors, i) {
			const int* v1 = vectors[i];
			const unsigned int label = labels[i];
			for (unsigned int j = i + 1; j < num_vectors; j++) {
				if (labels[j] != label) continue;
				const int* v2 = vectors[j];
				if (sparse_similarity(v1, v2) > DUPE_THRESHOLD) {
					al_add(kmeans_dups, i);
					al_add(kmeans_dups, j);
				}
			}
		}
	);
	
	fprintf(kmeans_file, "Checked!\n");
	fflush(kmeans_file);
	
	// Lock results.
	al_lock(kmeans_dups);
	al_trim_to_size(kmeans_dups);
	
	// Log duplocates found by the kmeans strategy.
	// fprintf(kmeans_file, "Duplocates found:\n");
	// size_t num_kmeans_dups = kmeans_dups->size;
	// for (size_t i = 0; i < num_kmeans_dups;) {
	// 	unsigned int d1 = al_get(kmeans_dups, i++);
	// 	unsigned int d2 = al_get(kmeans_dups, i++);
	// 	fprintf(kmeans_file, "%s (#%d) & %s (#%d)\n", dataset[d1], d1, dataset[d2], d2);
		
	// 	if (!verify_dupe(complete_dups, d1, d2)) {
	// 		printf("kmeans found false dupe: %s (#%d) & %s (#%d)\n", dataset[d1], d1, dataset[d2], d2);
	// 	}
	// }
	
	// Print kmeans clusters.
	// fprintf(kmeans_file, "\nPoints By Cluster Assignment:\n");
	// for (int cluster = 0; cluster < num_clusters; cluster++) {
	// 	fprintf(kmeans_file, "Cluster %d: ", cluster);
	// 	for (unsigned int i = 0u; i < NUM_VECTORS; i++) {
	// 		if (labels[i] == cluster) {
	// 			fprintf(kmeans_file, "%s, ", dataset[i]);
	// 		}
	// 	}
	// 	fprintf(kmeans_file, "\n");
	// }
	
	// Print cluster centroids. "my code is self-documenting"
	// size_t cur = 0;
	// char buf[num_clusters * (16 + NUM_DIMS * 6)];
	// cur += (size_t)snprintf(memset(buf, 0, sizeof(buf)), sizeof(buf) - cur, "\nFinal Centroids:\n");
	// for (int j = 0; j < num_clusters; j++) {
	// 	cur += (size_t)snprintf(buf + cur, sizeof(buf) - cur, "Cluster %d: (", j);
	// 	for (int dim = 0; dim < NUM_DIMS; dim++) {
	// 		double val = centroids[j][dim];
	// 		if (val > 0) {
	// 			if (val >= 0.0001) cur += (size_t)snprintf(buf + cur, sizeof(buf) - cur, "%.4lf", val);
	// 			else cur += (size_t)snprintf(buf + cur, sizeof(buf) - cur, "%.4lfe-9", val * 1000 * 1000 * 1000);
	// 		}
	// 		cur += (size_t)snprintf(buf + cur, sizeof(buf) - cur, ",");
	// 	}
	// 	#pragma GCC diagnostic push
	// 	#pragma GCC diagnostic ignored "-Wsequence-point"
	// 	cur += (size_t)snprintf(buf + --cur, sizeof(buf) - cur, ")\n");
	// 	#pragma GCC diagnostic pop
	// }
	// fprintf(kmeans_file, "%s", buf); // Flush
	
	// Print benchmarks.
	timer_stop(timer_total);
	const double total_time = timer_get(timer_total);
	timer_print_cmp(timer_clustering, INDENT"Kmeans clustering", total_time);
	timer_print_cmp(timer_scan, INDENT"Kmeans checking", total_time);
	printf(INDENT"Kmeans debug time: %.4lfs.\n", debug_time);
	
	// Free memory.
	free(labels);
	repeat (num_clusters, i) {
		free(centroids[i]);
	}
	free(centroids);
	timer_free(timer_total);
	timer_free(timer_clustering);
	timer_free(timer_scan);
	
	return kmeans_dups;
}

ArrayList* test_complete_search(int** vectors, unsigned int num_vectors, Timer* timer) {
	// Flush buffers to reduce flush overhead during benchmark.
	printf("\nComplete search on %u records:\n", num_vectors);
	check(fflush(stdout), "fflush(stdout)");
	
	// Execute the complete search dupe detection.
	timer_benchmark(timer,
		ArrayList* complete_dups = find_complete_dups(vectors, num_vectors);
	);
	
	// Print complete search summary.
	printf(INDENT"Dups: %ld\n", complete_dups->size / 2);
	printf(INDENT"Time: %.4lfs\n", timer_store(timer));
	check(fflush(stdout), "fflush(stdout)");
	
	return complete_dups;
}

void test_sliding_search(int** vectors, unsigned int num_vectors, Timer* timer, ArrayList* complete_dups) {
	// Sliding window with various window sizes.
	size_t num_window_sizes = sizeof(window_sizes) / sizeof(window_sizes[0]);
	repeat (num_window_sizes, i) {
		unsigned int window_size = window_sizes[i];
		
		// Flush buffers to reduce flush overhead during benchmark.
		printf("\nSliding window (x%u):\n", window_size);
		check(fflush(stdout), "fflush(stdout)");
			
		// Execute sliding window dupe detection.
		timer_benchmark(timer,
			ArrayList* sliding_dups = find_sliding_dups(vectors, num_vectors, window_size);
		);
		double sliding_time = timer_get(timer);
		
		// Print sliding summary.
		double percent_success_sliding = 100.0 * (double)sliding_dups->size / (double)complete_dups->size;
		printf(INDENT"Dups: %ld/%ld (%%%.2lf)\n", sliding_dups->size / 2, complete_dups->size / 2, percent_success_sliding);
		const double time_percent = 100.0f * sliding_time / timer->stored_duration;
		printf(INDENT"Time: %.4lfs (%%%.2f)\n", sliding_time, time_percent);
		check(fflush(stdout), "fflush(stdout)");
		
		// Free memory.
		al_free(sliding_dups);
	}
}

void test_kmeans_search(int** vectors, unsigned int num_vectors, unsigned int* cluster_counts, unsigned int num_cluster_counts, Timer* timer, ArrayList* complete_dups) {
	// Kmeans with various k values.
	repeat (num_cluster_counts, i) {
		unsigned int cluster_count = cluster_counts[i], iterations = 0;
		
		// Flush buffers to reduce flush overhead during benchmark.
		fprintf(kmeans_file, "\n\n============================\n%u vectors, %u clusters\n", num_vectors, cluster_count);
		printf("\nKmeans (num_clusters=%u):\n", cluster_count);
		check(fflush(stdout), "fflush(stdout)");
		
		// Execute kmeans dupe detection.
		timer_benchmark(timer,
			ArrayList* kmeans_dups = find_kmeans_dups(vectors, num_vectors, &iterations, max_iter, cluster_count);
		);
		double kmeans_time = timer_get(timer);
	
		// Print sliding summary.
		const double time_percent = 100.0f * kmeans_time / timer->stored_duration;
		const double percent_success_kmeans = 100.0 * (double)kmeans_dups->size / (2.0 * (double)complete_dups->size / 2);
		printf(INDENT"Time: %.4lfs (%%%.2f)\n", kmeans_time, time_percent);
		printf(INDENT"Iterations: %u/%u\n", iterations, max_iter);
		printf(INDENT"Dups: %ld/%ld (%%%.2lf)\n", kmeans_dups->size / 2, complete_dups->size / 2, percent_success_kmeans);
		check(fflush(stdout), "fflush(stdout)");
		
		// Free memory.
		al_free(kmeans_dups);
	}
}

// 100,000 records should take almost 10 minutes on LightSys hardware.
// Any more than 10 minutes, and we go to kmeans for faster execution time.
#define MAX_COMPLETE_SEARCH 0 // 100 * 1000
void test_lightning_search(int** vectors, unsigned int num_vectors, Timer* timer, ArrayList* complete_dups) {
	printf("\nLightning Search:\n");
	printf(INDENT"Data Size: %u\n", num_vectors);
	
	ArrayList* dups;
	if (num_vectors <= MAX_COMPLETE_SEARCH) {
		check(fflush(stdout), "fflush(stdout)");
		timer_benchmark(timer,
			dups = find_complete_dups(vectors, num_vectors);
		);
	} else {
		unsigned int cluster_count = compute_k(num_vectors), iterations = 0;
		printf(INDENT"Clusters: %u\n", cluster_count);
		check(fflush(stdout), "fflush(stdout)");
		
		// Execute kmeans dupe detection.
		timer_benchmark(timer,
			dups = find_kmeans_dups(vectors, num_vectors, &iterations, max_iter, cluster_count);
		);
		printf(INDENT"Iterations: %u/%u\n", iterations, max_iter);
	}
	
	// Early cleanup
	const size_t num_dups = dups->size;
	al_free(dups);

	// Print summary.
	const double time = timer_get(timer);
	const double time_percent = 100.0f * time / timer->stored_duration;
	const double percent_success = 100.0 * (double)num_dups / (2.0 * (double)complete_dups->size / 2);
	printf(INDENT"Time: %.4lfs (%%%.2f)\n", time, time_percent);
	printf(INDENT"Dups: %ld/%ld (%%%.2lf)\n", num_dups / 2, complete_dups->size / 2, percent_success);
	check(fflush(stdout), "fflush(stdout)");
}

/*** Program entry point: runs various tests and reports results.
 *** 
 *** Usage:
 *** 
 ***  - argv[1] -> complete_file_name
 *** 
 ***  - argv[2] -> sliding_file_name
 *** 
 ***  - argv[3] -> kmeans_file_name
 ***
 *** The program:
 *** 
 ***  - opens the three output files,
 *** 
 ***  - loads the dataset and builds vectors,
 *** 
 ***  - runs: complete search, sliding window, and kmeans clustering,
 *** 
 ***  - prints timing and success statistics, then frees resources and exits.
 ***
 *** @param argc Number of command-line arguments; must be 4.
 *** @param argv Command-line argument vector.
 *** @returns `0` on success, `1` on incorrect invocation.
 ***/
int main(int argc, char* argv[]) {
	if (argc != 4) {
		fprintf(stderr, "Usage: %s <complete_file_name> <sliding_file_name> <kmeans_file_name>\n", argv[0]);
		return 1;
	}
	complete_file = fopen(argv[1], "w");
	sliding_file = fopen(argv[2], "w");
	kmeans_file = fopen(argv[3], "w");
	Timer* timer = timer_new();
	Timer* timer_total = timer_new();
	timer_start(timer_total);
	
	// Set buffers to only flush manually for more accurate performance evaluation.
	setvbuf(stdout, NULL, _IOFBF, (2 * 1000 * 1000));
	setvbuf(kmeans_file, NULL, _IOFBF, (4 * 1000 * 1000));
	printf("Begin!\n");
	
	// test();
	// return 0;
	
	// Print basic settings info.
	printf(
		"\nSettings:\n"
			INDENT"Dimensions: %u\n"
			INDENT"Dupe Threshold: %.4f\n",
		NUM_DIMS,
		DUPE_THRESHOLD
	);
	fprint_mem(stdout);
	check(fflush(stdout), "fflush(stdout)");
	
	size_t num_dataset_sizes = sizeof(dataset_sizes) / sizeof(dataset_sizes[0]);
	repeat (num_dataset_sizes, i) {
		unsigned int dataset_size = dataset_sizes[i];
		
		// Load dataset and build vectors.
		char** dataset = (char**)mallocs(dataset_size * sizeof(char*));
		char path[BUFSIZ];
		snprintf(path, sizeof(path), "datasets/surnames_%u.txt", dataset_size);
		
		timer_start(timer);
		load_dataset(path, dataset, dataset_size);
		int** vectors = (int**)mallocs(dataset_size * sizeof(int*));
		repeat (dataset_size, j) {
			vectors[j] = build_vector(dataset[j]);
		}
		timer_stop(timer);
		
		// Print
		printf("\n\nDataset Loaded (x%u):\n", dataset_size);
		timer_print(timer, INDENT"Loading");
		fprint_mem(stdout);
		check(fflush(stdout), "fflush(stdout)");
		
		// Complete search.
		ArrayList* complete_dups = (dataset_size <= 50000)
			? test_complete_search(vectors, dataset_size, timer)
			: al_new();

		// Sliding search.
		test_sliding_search(vectors, dataset_size, timer, complete_dups);
		
		// Lightning search.
		test_lightning_search(vectors, dataset_size, timer, complete_dups);
		
		// Clean up.
		repeat (dataset_size, j) {
			free(vectors[j]);
			free(dataset[j]);
		}
		free(vectors);
		free(dataset);
		al_free(complete_dups);
		check(fflush(stdout), "fflush(stdout)");
	}
	
	// Print the total execution time.
	timer_stop(timer_total);
	timer_print(timer_total, "\nTotal");
	
	// End program and flush all buffers.
	printf("\nDone!\n");
	check(fflush(stdout), "fflush(stdout)");
	
	// Close files.
	if (complete_file) fclose(complete_file);
	if (sliding_file) fclose(sliding_file);
	if (kmeans_file) fclose(kmeans_file);
	
	// Free memory.
	timer_free(timer);
	timer_free(timer_total);
	
	return 0;
}
