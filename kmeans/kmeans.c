#include <ctype.h>
#include <float.h>
#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "lib/arraylist.h"
#include "lib/timer.h"
#include "lib/utils.h"

// Define the dataset.
// #define DATASET_PATH "datasets/surnames.txt"
// #define DATASET_SIZE 1000 * 1000
// #define DATASET_PATH "datasets/word_scrape.txt"
// #define DATASET_SIZE 19734
// #define DATASET_PATH "datasets/blackholes.txt"
// #define DATASET_SIZE 4632
// #define DATASET_PATH "datasets/dataset_unique_sorted.txt"
// #define DATASET_SIZE 1257

#define INDENT "\t> "
#define NUM_DIMS 251
#define SEED 1621963727
#define THRESHOLD 0.75
#define KMEANS_IMPROVEMENT_THRESHOLD 0.0010

// Test Parameters
unsigned int window_sizes[] = {3, 6, 16, 32, 64, 256};
unsigned int cluster_counts[] = {2, 4, 8, 16, 32, 64, 128, 256, 512, 1024};
unsigned int dataset_sizes[] = {5000, 10000};//, 50000, 100000, 1000000};
unsigned int max_iter = 64;
unsigned int num_repeats = 1; // Repeat algorithms to reduce randomness.

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

 
void fprint_vector(FILE* file, const double* vector) {
	for (unsigned int j = 0, idx = 0; idx < NUM_DIMS; j++) {
		if (vector[j] >= 0.001) { fprintf(file, " % 3g,", vector[j]); idx++; }
		else {
			unsigned int num = (unsigned int)(-vector[j]);
			idx += num;
			repeat(num, k) fprintf(file, "    ,");
		}
	}
	fprintf(file, "\n");
}

// ====================================
// Centralix Code

/*
 * hash_char_pair
 * This method creates an vector table index based a given character pair.
 * The characters are represented as their ASCII code points.
 *
 * Parameters:
 * 	num1 : first ASCII code point (double)
 * 	num2 : second ASCII code point (double)
 *
 * Returns:
 * 	vector table index (integer)
 * 
 * Assumptions
 * 	Both arguments are positive.
 */
unsigned int exp_fn_i_hash_char_pair(double num1, double num2) {
	unsigned int hash = (unsigned int)(round(((num1 * num1 * num1) + (num2 * num2 * num2)) * ((num1 + 1) / (num2 + 1)))) - 1;
	return hash % NUM_DIMS;
}

/*
 * exp_fn_i_dot_product
 * This method calculautes the dot product of two vectors.
 *
 * Parameters:
 * 	dot_product : the place where the result is stored (double)
 * 	r_freq_table1 : the first vector (double)
 * 	r_freq_table2 : the second vector (double)
 *
 * Returns:
 * 	0
 * 
 * Complexity: `O(d)`
 * 
 * LINK ../../centrallix-sysdoc/string_comparison.md#exp_fn_i_dot_product
 */
int exp_fn_i_dot_product(double* dot_product_ptr, const double* r_freq_table1, const double* r_freq_table2) {
	double dot_product = *dot_product_ptr;
	for (int i = 0; i < NUM_DIMS; i++) {
		dot_product += r_freq_table1[i] * r_freq_table2[i];
	}
	*dot_product_ptr = dot_product;
	return 0;
}

/*
 * exp_fn_i_magnitude
 * This method calculates the magnitude (also known as the normalization) of a vector
 * This is calculated as the square root of all squared elements in the vector
 *
 * Parameters:
 * 	magnitude : the place where the result is stored (double)
 * 	r_freq_table : the vector (double)
 * 
 * Complexity: `O(d)`
 * 
 * LINK ../../centrallix-sysdoc/string_comparison.md#exp_fn_i_magnitude
 */
int exp_fn_i_magnitude(double* magnitude_ptr, const double* r_freq_table) {
	double magnitude = *magnitude_ptr;
	for (int i = 0; i < NUM_DIMS; i++) {
		magnitude += r_freq_table[i] * r_freq_table[i];
	}
	*magnitude_ptr = sqrt(magnitude);
	return 0;
}

/*
 * exp_fn_i_frequency_table
 * This method creates a vector frequency table based on a string of characters.
 * This is essentially the hashing algorithm for a string into a vector
 *
 * Parameters:
 * 	table : integer pointer to vector frequency table (double)
 * 	term : the string of characters (char*)
 *
 * Returns:
 * 	0
 * 
 * LINK ../../centrallix-sysdoc/string_comparison.md#exp_fn_i_frequency_table
 */
int exp_fn_i_frequency_table(double* table, char* str) {
	// Initialize hash table with 0 values
	for (size_t i = 0; i < NUM_DIMS; i++) table[i] = 0;
	
	// j is the former character, i is the latter.
	int num_strs = (int)strlen(str);
	for (int j = -1, i = 0; i <= (int)num_strs; i++) {
		// If latter character is punctuation or whitespace, skip it.
		if (ispunct(str[i]) || isspace(str[i])) continue;

		unsigned int temp1 = (j == -1) ? '`' : (unsigned int)tolower(str[j]);
		unsigned int temp2 = (i == num_strs) ? '`' : (unsigned int)tolower(str[i]);

		// If either character is a number, reassign the code point
		// The significance of 75 here seems to be that it puts the numbers
		// right after the end of the lowercase letters, but they will still
		// colide with the {, |, }, ~ and DEL ASCII characters.
		if ('0' <= temp1 && temp1 <= '9') temp1 += 75;
		if ('0' <= temp2 && temp2 <= '9') temp2 += 75;
		
		// Hash the character pair into an index.
		unsigned int index = exp_fn_i_hash_char_pair((double)temp1, (double)temp2);
		
		// Increment Frequency Table value by number from 1 to 13.
		table[index] += (temp1 + temp2) % 13 + 1;
		
		j = i;
	}

	return 0;
}

// ====================================
// Research Code

/*** Helper function for compact error handling on library & system function calls.
 ***
 *** @param result The result of the function we're checking.
 *** @param functionName The name of the function being checked (for debugging).
 ***/
void check(const int result, const char* functionName) {
	if (result != 0) { // An error occured.
		// Create the most descriptive error message we can.
		char error_buf[BUFSIZ];
		snprintf(error_buf, sizeof(error_buf), "kmeans.c: Fail - %s", functionName);
		perror(error_buf);
		
		// Exit repeatedly until it works, in case exit gets interupted somehow.
		while (1) exit(result);
	}
}

/*** Allocates memory for vector storage.
 ***
 *** @returns A pointer to the new vector.
 ***/
double* create_vector(void) {
	double* vector = malloc(NUM_DIMS * sizeof(double));
	if (vector == NULL) {
		perror("Memory allocation failed.\n");
		while (true) exit(-1);
	}
	return vector;
}

/*** Takes an array of strings (strs) and converts them to vectors.
 *** create_vector() is called to allocate memeory for the new vectors.
 ***
 *** @param vectors The location to store new vectors will be stored.
 *** @param strs The strings used to build the vectors.
 *** @param num_vectors The number of vectors to create.
 *** @returns 0, success
 ***/
int build_vectors(double** vectors, char** strs, unsigned int num_vectors) {
	for (unsigned int i = 0; i < num_vectors; i++) {
		// Build dense vectors.
		double* vector = create_vector();
		check(exp_fn_i_frequency_table(vector, strs[i]), "exp_fn_i_frequency_table");
		
		// Count how much space is needed for a sparse vector.
		bool zero_prev = false;
		unsigned int size = 0;
		repeat(NUM_DIMS, j) {
			if (vector[j] < 0.001) {
				size += (zero_prev) ? 0 : 1;
				zero_prev = true;
			} else {
				size++;
				zero_prev = false;
			}
		}

		// Convert them to sparse vectors.
    	double* sparse = vectors[i] = (double*)malloc(size * sizeof(double));
    	if (!sparse) fprintf(stderr, "malloc(%u) - FAIL", size);
		size_t j = 0, sparse_idx = 0;
		while (j < NUM_DIMS) {
			if (vector[j] <= 0.001) {
				// Count and store consecutive zeros.
				size_t zero_count = 0;
				while (j < NUM_DIMS && vector[j] < 0.001) {
					zero_count++;
					j++;
				}
				sparse[sparse_idx++] = -(double)zero_count;
			} else {
				// Store the value
				sparse[sparse_idx++] = vector[j++];
			}
		}
		
		// Free unused data.
		free(vector);
	}
	return 0;
}

/*** Compute the value for `k` (number of clusters), given a dataset of with
 *** a size of `n`.
 *** 
 *** The following table shows data sizes vs. our aproximate guess at the
 *** best cluster size vs. the actual cluster size picked by the function.
 *** 
 *** ```csv
 *** Data Size, Guess, Actual
 *** 10k,       16,    10
 *** 100k,      32,    29
 *** 1M,        64,    57
 *** ```
 *** 
 *** Notice that this function consistantly undershoots the guessed optimal
 *** values. This is intentional. Fewer clusters, while slightly slower, gives
 *** better detection accuracy. Thus, this function is designed to air on the
 *** lower side, especially for smaller data sizes where the loss in speed is
 *** likely to be neglegable.
 *** 
 *** This function is not intended for datasets smaller than (`n < ~2000`).
 *** These should be handled using complete search.
 *** 
 *** @param n The size of the dataset.
 *** @returns k, the number of clusters to use.
 ***/
unsigned int compute_k(unsigned int n) {
	return max(2u, (unsigned int)pow(log(n) / log(36), 3.1) - 8u);
}

#define is_neg(val) (val < -0.001)

double magnitude(const double* vector) {
	double magnitude = 0.0;
	for (unsigned int i = 0, dim = 0; dim < NUM_DIMS;) {
		const double val = vector[i++];
		
		// Negative val represents -val 0s in the array, so skip that many values. 
		if (is_neg(val)) dim += (unsigned int)(-val);
		
		// We have a value, so square it and add it to the magnitude.
		else {magnitude += val * val; dim++;}
	}
	return sqrt(magnitude);
}

double magnitude_dense(const double* vector) {
	double magnitude = 0.0;
	for (int i = 0; i < NUM_DIMS; i++) {
		magnitude += vector[i] * vector[i];
	}
	return sqrt(magnitude);
}

void parse_token(double token, size_t* remaining, double* value) {
	if (is_neg(token)) {
		// This run contains -token zeros.
		*remaining = (size_t)llround(-token);
		*value = 0.0;
	} else {
		// This run contains one value.
		*remaining = 1;
		*value = token;
	}
}

double check_sim(double sim, const char* note, const double* v1, const double* v2) {
	if (sim < 0.0 || 1.000000001 < sim) {
		fprintf(stdout, "Strange similarity%s %g.\n", note, sim);
		fprintf(stdout, "v1:\t"); fprint_vector(stdout, v1);
		fprintf(stdout, "v2:\t"); fprint_vector(stdout, v2);
		fprintf(stdout, "\n");
		fflush(stdout);
	}
	return sim; // Chain calling.
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
double sparse_similarity(const double* v1, const double* v2) {
	// Calculate dot product
	double val1 = 0.0, val2 = 0.0, dot_product = 0.0;
	size_t vec1_remaining = 0, vec2_remaining = 0;
	size_t dim = 0, i1 = 0, i2 = 0;
	while (dim < NUM_DIMS) {
		if (vec1_remaining == 0) parse_token(v1[i1++], &vec1_remaining, &val1);
		if (vec2_remaining == 0) parse_token(v2[i2++], &vec2_remaining, &val2);

		// Accumulate the dot_product. If either vector is 0 here,
		// the total is 0 and this statement does nothing.
		dot_product += val1 * val2;
		
		// Consume overlap from both runs.
		size_t overlap = min(vec1_remaining, vec2_remaining);
		vec1_remaining -= overlap;
		vec2_remaining -= overlap;
		dim += overlap;
	}
	
	// Optional optimization to speed up nonsimilar vectors.
	if (dot_product < 0.001) return 0.0;
	
	// Return the difference score.
	return check_sim(fabs(dot_product) / (magnitude(v1) * magnitude(v2)), "", v1, v2);
}
// #define sparse_dif(v1, v2) (1.0 - sparse_similarity(v1, v2))

double sparse_similarity_c(const double* v1, const double* c2) {
	// Calculate dot product
	double dot_product = 0.0;
	for (unsigned int i = 0, dim = 0; dim < NUM_DIMS;) {
		const double val = v1[i++];
		
		// Negative val represents -val 0s in the array, so skip that many values. 
		if (is_neg(val)) dim += (unsigned int)(-val);
		
		// We have a value, so square it and add it to the magnitude.
		else dot_product += val * c2[dim++];
	}
	
	// Return the difference score.
	return check_sim(fabs(dot_product) / (magnitude(v1) * magnitude_dense(c2)), "_c", v1, c2);
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
double print_cluster_size(double** vectors, unsigned int num_vectors, unsigned int* labels, double** centroids, unsigned int num_clusters, unsigned int iteration) {
	double cluster_sums[num_clusters];
	double noncluster_sums[num_clusters];
	unsigned int cluster_counts[num_clusters];
	memset(cluster_sums, 0, sizeof(cluster_sums));
	memset(noncluster_sums, 0, sizeof(noncluster_sums));
	memset(cluster_counts, 0, sizeof(cluster_counts));
	
	// Sum the difference from each vector to its cluster centroid.
	for (unsigned int i = 0; i < num_vectors; i++) {
		unsigned int label = labels[i];
		double* vector = vectors[i];
		cluster_sums[label] += sparse_dif_c(vector, centroids[label]);
		cluster_counts[label]++;
		
		for (unsigned int j = 0; j < num_clusters; j++) {
			if (j == label) continue;
			noncluster_sums[j] += sparse_dif_c(vector, centroids[j]);
		}
	}
	
	// Calculate the average difference per cluster and then the overall average.
	// fprintf(kmeans_file, "Cluster Sizes:\n");
	int valid_clusters = 0;
	double cluster_total = 0.0, noncluster_total = 0.0;
	double max_cluster_size = 0.0, min_cluster_size = 1.0;
	int max_cluster_label = -1, min_cluster_label = -1;
	for (unsigned int label = 0; label < num_clusters; label++) {
		unsigned int cluster_count = cluster_counts[label];
		if (cluster_count > 0) {
			double cluster_size = cluster_sums[label] / cluster_count;
			double noncluster_size = noncluster_sums[label] / (num_vectors - cluster_count);
			cluster_total += cluster_size;
			noncluster_total += noncluster_size;
			valid_clusters++;
			
			if (cluster_size > max_cluster_size) {
				max_cluster_size = cluster_size;
				max_cluster_label = (int)label;
			}
			if (cluster_size < min_cluster_size) {
				min_cluster_size = cluster_size;
				min_cluster_label = (int)label;
			}
			
			// fprintf(kmeans_file,
			// 	"> Cluster #%d (x%d): %.4lf (vs. %.4lf).\n",
			// 	label, cluster_count, cluster_size, noncluster_size
			// ); // Debug
		}
	}
	
	// Verify that there are valid clusters.
	if (valid_clusters <= 0) {
		printf("kmeans #%u: No valid clusters!\n", iteration);
		return 0.0;
	}
	
	// Calculate average sizes.
	double average_cluster_size = cluster_total / valid_clusters;
	double average_noncluster_size = noncluster_total / valid_clusters;
	
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
void kmeans(double** vectors, unsigned int num_vectors, unsigned int* labels, double** centroids, unsigned int* iterations, unsigned int max_iter, unsigned int num_clusters, double* debug_time) {
	// Select random vectors to use as the initial centroids.
	srand(SEED);
	for (unsigned int i = 0; i < num_clusters; i++) {
		// Pick a random vector.
		const unsigned int random_index = (unsigned int)rand() % num_vectors;
		// fprintf(kmeans_file, "Centroid %u starts at vector %u.\n", i, random_index); // Debug
		
		// Sparse copy the vector into a densely allocated centroid.
		double* centroid = centroids[i];
		double* vector = vectors[random_index];
		for (int i = 0, dim = 0; dim < NUM_DIMS;) {
			double token = vector[i++];
			if (token > 0.0) centroid[dim++] = token;
			else repeat(-token, j) centroid[dim++] = 0.0;
		}
		
		// print_difference(vectors, 0, random_index); // Debug
	}
	// fprintf(kmeans_file, "\n");
	
	// Allocate memory for new centroids
	double** new_centroids = malloc((size_t)num_clusters * sizeof(double*));
	for (unsigned int i = 0; i < num_clusters; i++) {
		new_centroids[i] = create_vector();
	}
	
	// Main loop
	double previous_average_cluster_size = 1.0;
	int* cluster_counts = malloc((size_t)num_clusters * sizeof(int));
	for (unsigned int iter = 0; iter < max_iter; iter++) {
		bool changed = false;
		
		// Reset new centroids.
		for (unsigned int i = 0; i < num_clusters; i++) {
			for (int dim = 0; dim < NUM_DIMS; dim++) {
				new_centroids[i][dim] = 0.0;
			}
			cluster_counts[i] = 0;
		}
		
		// Assign each point to the nearest centroid.
		for (unsigned int i = 0; i < num_vectors; i++) {
			double* vector = vectors[i];
			double min_dist = DBL_MAX;
			unsigned int best_centroid_label = 0;
			
			// Find nearest centroid.
			for (unsigned int j = 0; j < num_clusters; j++) {
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
			for (unsigned int i = 0, dim = 0; dim < NUM_DIMS;) {
				const double val = vector[i++];
				if (is_neg(val)) dim += (unsigned int)(-val);
				else best_centroid[dim++] += val;
			}
			cluster_counts[best_centroid_label]++;
		}
		
		// Save number of iterations (if needed).
		if (iterations != NULL) *iterations = iter + 1;
		
		// Stop if centroids didn't change.
		if (!changed) break;
		
		// Update centroids.
		for (unsigned int i = 0; i < num_clusters; i++) {
			if (cluster_counts[i] <= 0)  continue;
			for (unsigned int dim = 0; dim < NUM_DIMS; dim++) {
				centroids[i][dim] = new_centroids[i][dim] / cluster_counts[i];
			}
		}
		
		// Print cluster size for debugging.
		Timer* timer = timer_new();
		timer_benchmark(timer,
			double average_cluster_size = print_cluster_size(vectors, num_vectors, labels, centroids, num_clusters, iter);
		);
		*debug_time += timer_get(timer);
		timer_free(timer);
		fflush(stdout);
		
		// Is there enough improvement?
		double improvement = previous_average_cluster_size - average_cluster_size;
		fprintf(kmeans_file, INDENT"Improvement: %.4lf\n", improvement);
		fflush(kmeans_file);
		if (improvement < KMEANS_IMPROVEMENT_THRESHOLD) break;
		previous_average_cluster_size = average_cluster_size;
	}
	
	// Free memory.
	for (unsigned int i = 0; i < num_clusters; i++) {
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
void load_dataset(const char* dataset_path, char** dataset, unsigned int dataset_size) {
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

/*** Checks whether the ordered pair (d1, d2) exists in `complete_dups`.
 ***
 *** @param complete_dups An ArrayList containing pairs of duplicate indices.
 *** @param d1 First index of the pair to verify.
 *** @param d2 Second index of the pair to verify.
 *** @returns true if the exact pair exists, false otherwise.
 ***/
bool verify_dupe(ArrayList* complete_dups, unsigned int d1, unsigned int d2) {
	return true;
	size_t num_complete_dups = complete_dups->size;
	for (size_t j = 0; j < num_complete_dups;) {
		unsigned int dc1 = al_get(complete_dups, j++);
		unsigned int dc2 = al_get(complete_dups, j++);
		if (dc1 == d1 && dc2 == d2) return true;
	}
	return false;
}

/*** Scans the entire dataset for duplicates by pairwise similarity.
 *** 
 *** This is the "complete" strategy: every pair `(i, j)` is compared and
 *** pairs with `(similarity > THRESHOLD)` are appended to the returned list.
 *** The function also logs found duplicates to the `complete_file`.
 *** 
 *** @param vectors Array of precomputed frequency vectors for all dataset strings.
 *** @returns A locked ArrayList containing duplicate index pairs.
 *** 
 *** Complexity: `O(n^2 * 3d)`
 ***/
ArrayList* find_complete_dups(double** vectors, unsigned int num_vectors, char** dataset) {
	ArrayList* complete_dups = al_newc(512);
	for (unsigned int i = 0; i < num_vectors; i++) {
		const double* v1 = vectors[i];
		for (unsigned int j = i + 1; j < num_vectors; j++) {
			const double* v2 = vectors[j];
			if (sparse_similarity(v1, v2) > THRESHOLD) {
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
 *** @param window_size The size of the sliding window.
 *** @param complete_dups The complete-dups list used to validate results.
 *** @returns A locked ArrayList containing duplicate index pairs.
 *** 
 *** Complexity: `O(nw * 3d)`
 ***/
ArrayList* find_sliding_dups(double** vectors, unsigned int num_vectors, unsigned int window_size, ArrayList* complete_dups) {
	ArrayList* sliding_dups = al_newc(512);
	for (unsigned int i = 0; i < num_vectors; i++) {
		const double* v1 = vectors[i];
		const unsigned int j_max = min(i + window_size, num_vectors);
		for (unsigned int j = i + 1; j < j_max; j++) {
			const double* v2 = vectors[j];
			if (sparse_similarity(v1, v2) > THRESHOLD) {
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
 ***    with `(similarity > THRESHOLD)`.
 *** 
 ***  - Log duplicates and cluster contents to `kmeans_file`, validate found
 ***    pairs against `complete_dups` to detect mistakes.
 ***
 *** @param vectors Array of precomputed frequency vectors for all dataset strings.
 *** @param max_iter Maximum iterations passed to kmeans().
 *** @param num_clusters Number of clusters to produce.
 *** @param complete_dups The complete-dups list used to validate results.
 *** @returns A locked ArrayList containing duplicate index pairs.
 *** 
 *** Complexity: `O((nk + nd) + 3d*s^2)`
 *** Complexity: `O((nk + nd) + 3d*(n/k)^2)`
 *** Complexity: `O(nk + (n/k)^2)` (ignoring i & d)
 ***/
ArrayList* find_kmeans_dups(double** vectors, unsigned int num_vectors, unsigned int* iterations, unsigned int max_iter, unsigned int num_clusters, ArrayList* complete_dups) {
	// Create timers.
	Timer* timer_total = timer_new();
	Timer* timer_clustering = timer_new();
	Timer* timer_scan = timer_new();
	timer_start(timer_total);
	
	// Malloc memory for finding clusters.
	unsigned int* labels = malloc(num_vectors * sizeof(unsigned int));
	for (unsigned int i = 0; i < num_vectors; i++) labels[i] = 0;
	double** centroids = malloc((size_t)num_clusters * sizeof(double*));
	for (unsigned int i = 0; i < num_clusters; i++)
		centroids[i] = malloc(NUM_DIMS * sizeof(double));
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
		for (unsigned int i = 0; i < num_vectors; i++) {
			const double* v1 = vectors[i];
			const unsigned int label = labels[i];
			for (unsigned int j = i + 1; j < num_vectors; j++) {
				if (labels[j] != label) continue;
				const double* v2 = vectors[j];
				if (sparse_similarity(v1, v2) > THRESHOLD) {
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
	// 	for (unsigned int i = 0; i < NUM_VECTORS; i++) {
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
	double total_time = timer_get(timer_total);
	timer_print_cmp(timer_clustering, INDENT"Kmeans clustering", total_time);
	timer_print_cmp(timer_scan, INDENT"Kmeans checking", total_time);
	printf(INDENT"Kmeans debug time: %.4lfs.\n", debug_time);
	
	// Free memory.
	free(labels);
	for (unsigned int i = 0; i < num_clusters; i++) {
		free(centroids[i]);
	}
	free(centroids);
	timer_free(timer_total);
	timer_free(timer_clustering);
	timer_free(timer_scan);
	
	return kmeans_dups;
}

ArrayList* test_complete_search(double** vectors, unsigned int num_vectors, Timer* timer, char** dataset) {
	// Flush buffers to reduce flush overhead during benchmark.
	printf("\nComplete search on %u records:\n", num_vectors);
	check(fflush(stdout), "fflush(stdout)");
	
	// Execute the complete search dupe detection.
	timer_benchmark(timer,
		ArrayList* complete_dups = find_complete_dups(vectors, num_vectors, dataset);
	);
	
	// Print complete search summary.
	printf(INDENT"Dups: %ld\n", complete_dups->size / 2);
	printf(INDENT"Time: %.4lfs\n", timer_store(timer));
	check(fflush(stdout), "fflush(stdout)");
	
	return complete_dups;
}

void test_sliding_search(double** vectors, unsigned int num_vectors, Timer* timer, ArrayList* complete_dups) {
	// Sliding window with various window sizes.
	size_t num_window_sizes_count = sizeof(window_sizes) / sizeof(window_sizes[0]);
	for (size_t i = 0; i < num_window_sizes_count; i++) {
		unsigned int window_size = window_sizes[i];
		
		// Flush buffers to reduce flush overhead during benchmark.
		printf("\nSliding window (x%u):\n", window_size);
		check(fflush(stdout), "fflush(stdout)");
		
		// Benchmarking execution time.
		double sliding_time = 0.0;
		ArrayList* sliding_dups = NULL;
		for (unsigned int rep = 0; rep < num_repeats; rep++) {
			// Free memory.
			if (sliding_dups != NULL) al_free(sliding_dups);
			
			// Execute sliding window dupe detection.
			timer_benchmark(timer,
				sliding_dups = find_sliding_dups(vectors, num_vectors, window_size, complete_dups);
			);
			sliding_time += timer_get(timer);
		}
		sliding_time /= num_repeats;
		
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

void test_kmeans_search(double** vectors, unsigned int num_vectors, unsigned int* cluster_counts, unsigned int num_cluster_counts, Timer* timer, ArrayList* complete_dups) {
	// Kmeans with various k values.
	for (size_t i = 0; i < num_cluster_counts; i++) {
		unsigned int cluster_count = cluster_counts[i], iterations = 0;
		
		// Flush buffers to reduce flush overhead during benchmark.
		fprintf(kmeans_file, "\n\n============================\n%u vectors, %u clusters\n", num_vectors, cluster_count);
		printf("\nKmeans (num_clusters=%u):\n", cluster_count);
		check(fflush(stdout), "fflush(stdout)");
		
		// Benchmarking execution time.
		double kmeans_time = 0.0;
		ArrayList* kmeans_dups = NULL;
		for (unsigned int rep = 0; rep < num_repeats; rep++) {
			// Free memory.
			if (kmeans_dups != NULL) al_free(kmeans_dups);
			
			// Execute kmeans dupe detection.
			timer_benchmark(timer,
				kmeans_dups = find_kmeans_dups(vectors, num_vectors, &iterations, max_iter, cluster_count, complete_dups);
			);
			kmeans_time += timer_get(timer);
		}
		kmeans_time /= num_repeats;
	
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

#define MAX_COMPLETE_SEARCH 1000
void test_lightning_search(double** vectors, unsigned int num_vectors, Timer* timer, ArrayList* complete_dups, char** dataset) {
	printf("\nLightning Search:\n");
	printf(INDENT"Data Size: %u\n", num_vectors);
	
	ArrayList* dups;
	if (num_vectors <= MAX_COMPLETE_SEARCH) {
		check(fflush(stdout), "fflush(stdout)");
		timer_benchmark(timer,
			dups = find_complete_dups(vectors, num_vectors, dataset);
		);
	} else {
		unsigned int cluster_count = compute_k(num_vectors), iterations = 0;
		printf(INDENT"Clusters: %u\n", cluster_count);
		check(fflush(stdout), "fflush(stdout)");
		
		// Execute kmeans dupe detection.
		timer_benchmark(timer,
			dups = find_kmeans_dups(vectors, num_vectors, &iterations, max_iter, cluster_count, complete_dups);
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

// Data random sample.
// double** sample_vectors(double** vectors, unsigned int dataset_size) {
//     // Create an array of indices
//     unsigned int* indices = (unsigned int*)malloc(DATASET_SIZE * sizeof(unsigned int));
//     for (unsigned int i = 0; i < DATASET_SIZE; i++) {
//         indices[i] = i;
//     }

//     // Shuffle the indices array using Fisher-Yates shuffle
//     for (unsigned int i = DATASET_SIZE - 1; i > 0; i--) {
//         unsigned int j = (unsigned int)rand() % (i + 1u);
//         unsigned int temp = indices[i];
//         indices[i] = indices[j];
//         indices[j] = temp;
//     }

//     // Select the first dataset_size indices as the sample
// 	double** sample = (double**)malloc(dataset_size * sizeof(double*));
//     for (unsigned int i = 0; i < dataset_size; i++) {
//         sample[i] = vectors[indices[i]];
//     }

// 	// Clean up.
//     free(indices);
	
// 	return sample;
// }

// void test(void) {
// 	char* dataset[] = {"box", "boxes", "school"};
// 	unsigned int dataset_size = 3u;
// 	double** vectors = malloc(dataset_size * sizeof(double*));
// 	check(build_vectors(vectors, dataset, dataset_size), "build_vectors");
	
// 	repeat(dataset_size, i) {
// 		double* vector = vectors[i];
// 		printf("%s:", dataset[i]);
// 		for (unsigned int j = 0, idx = 0; idx < NUM_DIMS; j++) {
// 			if (vector[j] >= 0.001) idx++;
// 			else idx += (unsigned int)(-vector[j]);
// 			printf(" %g,", vector[j]);
// 		}
// 		printf("\n");
// 	}
	
// 	printf("\t\t");
// 	repeat(NUM_DIMS, i) printf("% 4d,", i);
// 	printf("\n");
	
// 	repeat(dataset_size, i) {
// 		double* vector = vectors[i];
// 		printf("%s:\t", dataset[i]);
// 		for (unsigned int j = 0, idx = 0; idx < NUM_DIMS; j++) {
// 			if (vector[j] >= 0.001) { printf(" % 3g,", vector[j]); idx++; }
// 			else {
// 				unsigned int num = (unsigned int)(-vector[j]);
// 				idx += num;
// 				repeat(num, k) printf("    ,");
// 			}
// 		}
// 		printf("\n");
// 	}
	
// 	for (unsigned int i = 0; i < dataset_size; i++) {
// 		const double* v1 = vectors[i];
// 		for (unsigned int j = i + 1; j < dataset_size; j++) {
// 			const double* v2 = vectors[j];
// 			printf("%s (#%u) & %s (#%u) => %lf\n", dataset[i], i, dataset[j], j, sparse_similarity(v1, v2));
// 		}
// 	}
// }

/*** Program entry point: runs three duplicate-detection strategies and reports results.
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
			INDENT"Similarity Threshold: %.4f\n",
		NUM_DIMS,
		THRESHOLD
	);
	check(fflush(stdout), "fflush(stdout)");
	
	size_t num_dataset_sizes = sizeof(dataset_sizes) / sizeof(dataset_sizes[0]);
	repeat(num_dataset_sizes, i) {
		unsigned int dataset_size = dataset_sizes[i];
		
		// Load dataset and build vectors.
		char* dataset[dataset_size];
		timer_benchmark(timer,
			char path[BUFSIZ];
			snprintf(path, sizeof(path), "datasets/surnames_%u.txt", dataset_size);
			
			load_dataset(path, dataset, dataset_size);
			double** vectors = malloc(dataset_size * sizeof(double*));
			check(build_vectors(vectors, dataset, dataset_size), "build_vectors");
		);
		
		// Print
		printf("\n\nDataset Loaded (x%u):\n", dataset_size);
		timer_print(timer, INDENT"Loading");
		
		// Complete search.
		ArrayList* complete_dups = (dataset_size <= 10000)
			? test_complete_search(vectors, dataset_size, timer, dataset)
			: al_new();

		// Sliding search.
		// test_sliding_search(vectors, dataset_size, timer, complete_dups);
		
		// Kmeans search.
		// test_kmeans_search(vectors, dataset_size, timer, complete_dups);
		
		// Lightning search.
		test_lightning_search(vectors, dataset_size, timer, complete_dups, dataset);
		
		// Clean up.
		for (unsigned int i = 0; i < dataset_size; i++) {
			free(vectors[i]);
			free(dataset[i]);
		}
		free(vectors);
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
