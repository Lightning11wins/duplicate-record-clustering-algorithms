#include <ctype.h>
#include <float.h>
#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "lib/arraylist.h"

// Define the dataset.
#define DATASET_PATH "dataset_unique_sorted.txt"
#define DATASET_SIZE 1257 // 1975

#define MAX_ITER 16                 // Maximum iterations.
#define NUM_CLUSTERS 64            // Number of clusters (K).
#define NUM_DIMS 251               // The number of dimensions used for clustering data.
#define NUM_VECTORS DATASET_SIZE   // The number of strings in the test dataset.
#define SEED 0                     // The seed used for randomly selecting centroids.
#define THRESHOLD 0.75

char* dataset[NUM_VECTORS];
FILE* complete_file = NULL;
FILE* sliding_file = NULL;
FILE* kmeans_file = NULL;

#define streql(str1, str2) !strcmp((str1), (str2))

// Timer code
#define duration(start, end) ((end) - (start))
#define timer_init() double timer_start = -1, timer_end = -1, timer_duration = -1;
#define timer_start() timer_start = monotonic_seconds()
#define timer_stop() timer_end = monotonic_seconds()
#define timer_store() timer_duration = duration(timer_start, timer_end)
#define timer_print(name) print_time(name, duration(timer_start, timer_end))
#define timer_print_cmp(name) print_time_cmp(name, duration(timer_start, timer_end))
#define print_time(name, seconds) printf("\n%s time: %0.04fs.\n", name, seconds)
#define print_time_cmp(name, seconds) printf("\n%s time: %0.04fs (%%%.2lf).\n", name, seconds, 100.0 * (double)(seconds) / (double)timer_duration)

static double monotonic_seconds(void) {
	struct timespec ts;
	clock_gettime(CLOCK_MONOTONIC, &ts);
	return (double)ts.tv_sec + (double)ts.tv_nsec / 1e9;
}

#define max(a, b) ({ \
	__typeof__ (a) _a = (a); \
	__typeof__ (b) _b = (b); \
	(_a > _b) ? _a : _b; \
})

#define min(a, b) ({ \
	__typeof__ (a) _a = (a); \
	__typeof__ (b) _b = (b); \
	(_a < _b) ? _a : _b; \
})

// Things to fix:
// Optimize Data Conversions
// Add documentation
// Free memory

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
static int exp_fn_i_hash_char_pair(double num1, double num2) {
	int hash = (int)round(((num1 * num1 * num1) + (num2 * num2 * num2)) * ((num1 + 1) / (num2 + 1))) - 1;
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
 * LINK ../../centrallix-sysdoc/string_comparison.md#exp_fn_i_dot_product
 */
static int exp_fn_i_dot_product(double* dot_product_ptr, const double* r_freq_table1, const double* r_freq_table2) {
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
 * LINK ../../centrallix-sysdoc/string_comparison.md#exp_fn_i_magnitude
 */
static int exp_fn_i_magnitude(double* magnitude_ptr, const double* r_freq_table) {
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
static int exp_fn_i_frequency_table(double* table, char* term) {
	// Initialize hash table with 0 values
	for (int i = 0; i < NUM_DIMS; i++) table[i] = 0;
	
	// j is the former character, i is the latter.
	int num_terms = (int)strlen(term);
	for (int j = -1, i = 0; i <= num_terms; j = i++) {
		// If latter character is punctuation or whitespace, skip it
		if (ispunct(term[i]) || isspace(term[i])) continue;

		double temp1 = (j == -1) ? '`' : (int)tolower(term[j]);
		double temp2 = (i == num_terms) ? '`' : (int)tolower(term[i]);

		// If either character is a number, reassign the code point
		// The significance of 75 here seems to be that it puts the numbers
		// right after the end of the lowercase letters, but they will still
		// colide with the {, |, }, ~ and DEL ASCII characters.
		if ('0' <= temp1 && temp1 <= '9') temp1 += 75;
		if ('0' <= temp2 && temp2 <= '9') temp2 += 75;
		
		// Hash the character pair into an index
		int index = exp_fn_i_hash_char_pair(temp1, temp2);
		
		// Increment Frequency Table value by number from 1 to 13
		table[index] += ((int)temp1 + (int)temp2) % 13 + 1;
		// table[index] += 1;
	}

	return 0;
}

// ====================================
// Vectoring Code

/*** Helper function for compact error handling on library & system function calls.
 ***
 *** @param result The result of the function we're checking.
 *** @param functionName The name of the function being checked (for debugging).
 ***/
static void check(const int result, const char* functionName) {
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
static double* create_vector(void) {
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
static int build_vectors(double** vectors, char** strs, size_t num_vectors) {
	for (size_t i = 0; i < num_vectors; i++) {
		double* vector = vectors[i] = create_vector();
		check(exp_fn_i_frequency_table(vector, strs[i]), "exp_fn_i_frequency_table");
	}
	return 0;
}

/*** Calculates the cosine similarity, aka. the angle between the two
 *** vectors in n dimensional space (where n is NUM_DIMS). In other words,
 *** `similarity(A, B)` performs the function `cos(θ) = (A. B)/(|A| |B|)`
 *** where A and B are vectors.
 ***
 *** Assumes all dimensions of both parameters are assumed to be strictly positive.
 ***
 *** @param v1 The first vector being compared.
 *** @param v2 The second vector being compared.
 *** @returns {0 - 1} where 0 indicates that the vectors (and the strings they
 ***          represent) have no similarity and 1 indicates that they are identical.
 ***/
static double similarity(const double* v1, const double* v2) {
	// Calculate dot product
	double dot_product = 0;
	check(exp_fn_i_dot_product(&dot_product, v1, v2), "exp_fn_i_dot_product");
	
	// Calculate magnitudes of each relative frequency vector
	double magnitude1 = 0;
	double magnitude2 = 0;
	check(exp_fn_i_magnitude(&magnitude1, v1), "exp_fn_i_magnitude");
	check(exp_fn_i_magnitude(&magnitude2, v2), "exp_fn_i_magnitude");

	// Calculate the similarity score
	return fabs(dot_product) / (magnitude1 * magnitude2);
}

/*** Inverts the cosine similarity such that more similar vectors produce a
 *** smaller value instead of a larger value.
 ***
 *** Assumes all dimensions of both parameters are assumed to be strictly positive.
 ***
 *** @param v1 The first vector being compared.
 *** @param v2 The second vector being compared.
 *** @returns {0 - 1}: where 0 indicates that the vectors (and the strings they
 ***          represent) are identical and 1 indicates that they have no similarity.
 ***/
#define difference(v1, v2) (1.0 - similarity((v1), (v2)))

// Debug helper function to print differences.
#define print_difference(vectors, i1, i2)                              \
	printf(                                                            \
		"Difference from '%s' to '%s': %lf\n",                         \
		dataset[i1], dataset[i2], difference(vectors[i1], vectors[i2]) \
	); // Debug

/*** Calculate the average size of all clusters in a set of vectors.
 ***
 *** @param vectors The vectors of the dataset.
 *** @param num_vectors The number of vectors in the dataset.
 *** @param labels The clusters to which vectors are assigned.
 *** @param centroids The locations of the centroids.
 *** @returns 0, success
 ***/
static int print_cluster_size(double** vectors, int num_vectors, int* labels, double** centroids, int iteration) {
	double cluster_sums[NUM_CLUSTERS] = {0};
	double noncluster_sums[NUM_CLUSTERS] = {0};
	int cluster_counts[NUM_CLUSTERS] = {0};
	
	// Sum the difference from each vector to its cluster centroid.
	for (int i = 0; i < num_vectors; i++) {
		int label = labels[i];
		double* vector = vectors[i];
		cluster_sums[label] += difference(vector, centroids[label]);
		cluster_counts[label]++;
		
		for (int j = 0; j < NUM_CLUSTERS; j++) {
			if (j == label) continue;
			noncluster_sums[j] += difference(vector, centroids[j]);
		}
	}
	
	// Calculate the average difference per cluster and then the overall average.
	fprintf(kmeans_file, "\nCluster Sizes:\n");
	int valid_clusters = 0;
	double cluster_total = 0.0, noncluster_total = 0.0;
	double max_cluster_size = 0.0, min_cluster_size = 1.0;
	int max_cluster_label = -1, min_cluster_label = -1;
	for (int label = 0; label < NUM_CLUSTERS; label++) {
		int cluster_count = cluster_counts[label];
		if (cluster_count > 0) {
			double cluster_size = cluster_sums[label] / cluster_count;
			double noncluster_size = noncluster_sums[label] / (num_vectors - cluster_count);
			cluster_total += cluster_size;
			noncluster_total += noncluster_size;
			valid_clusters++;
			
			if (cluster_size > max_cluster_size) {
				max_cluster_size = cluster_size;
				max_cluster_label = label;
			}
			if (cluster_size < min_cluster_size) {
				min_cluster_size = cluster_size;
				min_cluster_label = label;
			}
			
			fprintf(kmeans_file,
				"> Cluster #%d (x%d): %.4lf (vs. %.4lf).\n",
				label, cluster_count, cluster_size, noncluster_size
			); // Debug
		}
	}
	
	// Final print
	if (valid_clusters > 0) {
		double average_cluster_size = cluster_total / valid_clusters;
		double average_noncluster_size = noncluster_total / valid_clusters;
		
		printf(
			"\nkmeans #%d:\n"
				"\t> Average cluster: %.4lf\n"
				"\t> Average noncluster: %.4lf\n"
				"\t> Largest cluster: #%d @ %.4lf\n"
				"\t> Smallest cluster: #%d @ %.4lf\n",
			iteration,
			average_cluster_size,
			average_noncluster_size,
			max_cluster_label, max_cluster_size,
			min_cluster_label, min_cluster_size
		);
	} else printf("kmeans #%d: No valid clusters!\n", iteration);
	
	return 0;
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
 ***/
static void kmeans(double** vectors, int num_vectors, int* labels, double** centroids) {
	// Select random vectors to use as the initial centroids.
	srand(SEED);
	for (int i = 0; i < NUM_CLUSTERS; i++) {
		// Pick a random vector.
		const int random_index = rand() % num_vectors;
		fprintf(kmeans_file, "Centroid %d starts at vector %d.\n", i, random_index); // Debug
		
		// Copy each dimetion from the selected random vector to the current centroid.
		for (int dim = 0; dim < NUM_DIMS; dim++) {
			centroids[i][dim] = vectors[random_index][dim];
		}
		
		// print_difference(vectors, 0, random_index); // Debug
	}
	
	// Allocate memory for new centroids
	double** new_centroids = malloc(NUM_CLUSTERS * sizeof(double*));
	for (int i = 0; i < NUM_CLUSTERS; i++) {
		new_centroids[i] = create_vector();
	}
	
	// Main loop
	int* cluster_counts = malloc(NUM_CLUSTERS * sizeof(int));
	for (int i = 0; i < MAX_ITER; i++) {
		bool changed = false;
		
		// Reset new centroids
		for (int i = 0; i < NUM_CLUSTERS; i++) {
			for (int dim = 0; dim < NUM_DIMS; dim++) {
				new_centroids[i][dim] = 0.0;
			}
			cluster_counts[i] = 0;
		}
		
		// Assign each point to the nearest centroid
		for (int i = 0; i < num_vectors; i++) {
			double* vector = vectors[i];
			double min_dist = DBL_MAX;
			int best_centroid_label = 0;
			
			for (int j = 0; j < NUM_CLUSTERS; j++) {
				double dist = difference(vector, centroids[j]);
				if (dist < min_dist) {
					min_dist = dist;
					best_centroid_label = j;
				}
			}
			
			if (labels[i] != best_centroid_label) {
				labels[i] = best_centroid_label;
				changed = true;
			}
			
			// Accumulate values for new centroid calculation
			double* best_centroid = new_centroids[best_centroid_label];
			for (int dim = 0; dim < NUM_DIMS; dim++) {
				best_centroid[dim] += vector[dim];
			}
			cluster_counts[best_centroid_label]++;
		}
		
		// Update centroids
		for (int j = 0; j < NUM_CLUSTERS; j++) {
			if (cluster_counts[j] > 0) {
				for (int dim = 0; dim < NUM_DIMS; dim++) {
					centroids[j][dim] = new_centroids[j][dim] / cluster_counts[j];
				}
			}
		}
		
		print_cluster_size(vectors, num_vectors, labels, centroids, i);
		
		// Stop if centroids didn't change.
		if (!changed) break;
	}
	
	// Final centroid update.
	for (int i = 0; i < num_vectors; i++) {
		double* vector = vectors[i];
		double min_dist = DBL_MAX;
		int best_centroid_label = 0;
		
		for (int j = 0; j < NUM_CLUSTERS; j++) {
			double dist = difference(vector, centroids[j]);
			if (dist < min_dist) {
				min_dist = dist;
				best_centroid_label = j;
			}
		}
		labels[i] = best_centroid_label;
	}
	
	// Free memory.
	for (int i = 0; i < NUM_CLUSTERS; i++) {
		free(new_centroids[i]);
	}
	free(new_centroids);
	free(cluster_counts);
}

// Not thread safe.
static void load_dataset(const char* dataset_path) {
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
	
	size_t count = 0;
	char* token = strtok(buffer, ",");
	while (token && count < NUM_VECTORS) {
		dataset[count++] = strdup(token);
		token = strtok(NULL, ",");
	}
	
	free(buffer);
}

int main(int argc, char* argv[]) {
	if (argc != 4) {
		fprintf(stderr, "Usage: %s <complete_file_name> <sliding_file_name> <kmeans_file_name>\n", argv[0]);
		return 1;
	}
	complete_file = fopen(argv[1] ,"w");
	sliding_file = fopen(argv[2] ,"w");
	kmeans_file = fopen(argv[3] ,"w");
	
	// Set buffers to only flush manually for more accurate performance evaluation.
	setvbuf(stdout, NULL, _IOFBF, (4 * 1000 * 1000));
	setvbuf(kmeans_file, NULL, _IOFBF, (4 * 1000 * 1000));
	printf("Begin!\n");
	check(fflush(stdout), "fflush(stdout)");
	
	// Load the dataset from the .gitignored dataset file.
	load_dataset(DATASET_PATH);
	
	// Setup timer.
	timer_init();
	
	// Allocate ram to store vectors.
	double** vectors = malloc(NUM_VECTORS * sizeof(double*));
	
	// Build the vectors.
	check(build_vectors(vectors, dataset, NUM_VECTORS), "build_vectors");
	
	// Execute the complete solution.
	timer_start();
	ArrayList* complete_dups = al_initc(512);
	for (int i = 0; i < NUM_VECTORS; i++) {
		const double* v1 = vectors[i];
		for (int j = i + 1; j < NUM_VECTORS; j++) {
			const double* v2 = vectors[j];
			if (similarity(v1, v2) > THRESHOLD) {
				al_add(complete_dups, i);
				al_add(complete_dups, j);
			}
		}
	}
	timer_stop();
	timer_print("Complete similarity");
	timer_store();
	
	// Lock result.
	al_trim_to_size(complete_dups);
	al_lock(complete_dups);
	
	// Print basic information.
	size_t num_complete_dups = complete_dups->size;
	printf("Complete search found %ld dups.\n", num_complete_dups / 2);
	check(fflush(stdout), "fflush(stdout)");
	
	// Log duplocates found by the complete strategy.
	fprintf(complete_file, "Duplocates found:\n");
	for (size_t i = 0; i < num_complete_dups;) {
		int d1 = al_get(complete_dups, i++), d2 = al_get(complete_dups, i++);
		fprintf(complete_file, "%s (#%d) & %s (#%d)\n", dataset[d1], d1, dataset[d2], d2);
	}
	
	// Execute current (sliding window) solution.
	timer_start();
	ArrayList* sliding_dups = al_initc(512);
	for (int i = 0; i < NUM_VECTORS; i++) {
		const double* v1 = vectors[i];
		const int j_max = min(i + 7, NUM_VECTORS);
		for (int j = i + 1; j < j_max; j++) {
			const double* v2 = vectors[j];
			if (similarity(v1, v2) > THRESHOLD) {
				al_add(sliding_dups, i);
				al_add(sliding_dups, j);
			}
		}
	}
	timer_stop();
	timer_print_cmp("Sliding similarity");
	
	// Lock result.
	al_trim_to_size(sliding_dups);
	al_lock(sliding_dups);
	
	// Print basic information.
	size_t num_sliding_dups = sliding_dups->size;
	printf("Sliding window found %ld dups (%%%.2lf).\n", num_sliding_dups / 2, 100.0 * (double)num_sliding_dups / (double)num_complete_dups);
	check(fflush(stdout), "fflush(stdout)");
	
	// Log duplocates found by the sliding strategy.
	fprintf(sliding_file, "Duplocates found:\n");
	for (size_t i = 0; i < num_sliding_dups;) {
		int d1 = al_get(sliding_dups, i++), d2 = al_get(sliding_dups, i++);
		fprintf(sliding_file, "%s (#%d) & %s (#%d)\n", dataset[d1], d1, dataset[d2], d2);
		
		// Verify dupes.
		int verified = 0;
		for (size_t j = 0; j < num_complete_dups;) {
			int dc1 = al_get(complete_dups, j++), dc2 = al_get(complete_dups, j++);
			if (dc1 == d1 && dc2 == d2) {
				verified = 1;
				break;
			}
		}
		if (!verified) {
			printf("sliding found false dupe between: %s (#%d) & %s (#%d)\n", dataset[d1], d1, dataset[d2], d2);
		}
	}
	
	// Start kmeans clustering timer.
	timer_start();
	
	// Allocate memory for clustering labels.
	int* labels = malloc(NUM_VECTORS * sizeof(int));
	for (int i = 0; i < NUM_VECTORS; i++) labels[i] = -1;
	
	// Allocate memory for clustering centroids.
    double** centroids = malloc(NUM_CLUSTERS * sizeof(double*));
	for (int i = 0; i < NUM_CLUSTERS; i++) {
		centroids[i] = malloc(NUM_DIMS * sizeof(double));
	}
	
	// Execute kmeans clustering.
	kmeans(vectors, NUM_VECTORS, labels, centroids);
	
	// Find duplocates in clusters.
	ArrayList* kmeans_dups = al_initc(512);
	for (int i = 0; i < NUM_VECTORS; i++) {
		const double* v1 = vectors[i];
		const int label = labels[i];
		for (int j = i + 1; j < NUM_VECTORS; j++) {
			if (labels[j] != label) continue;
			const double* v2 = vectors[j];
			if (similarity(v1, v2) > THRESHOLD) {
				al_add(kmeans_dups, i);
				al_add(kmeans_dups, j);
			}
		}
	}
	
	// Stop kmeans clustering timer.
	timer_stop();
	timer_print_cmp("kmeans clustering");
	
	// Lock result.
	al_trim_to_size(kmeans_dups);
	al_lock(kmeans_dups);
	
	// Print basic information.
	size_t num_kmeans_dups = kmeans_dups->size;
	printf("kmeans found %ld dups (%%%.2lf).\n", num_kmeans_dups / 2, 100.0 * (double)num_kmeans_dups / (double)num_complete_dups);
	check(fflush(stdout), "fflush(stdout)");
	
	// Log duplocates found by the kmeans strategy.
	fprintf(kmeans_file, "Duplocates found:\n");
	for (size_t i = 0; i < num_kmeans_dups;) {
		int d1 = al_get(kmeans_dups, i++), d2 = al_get(kmeans_dups, i++);
		fprintf(kmeans_file, "%s (#%d) & %s (#%d)\n", dataset[d1], d1, dataset[d2], d2);
		
		// Verify dupes.
		int verified = 0;
		for (size_t j = 0; j < num_complete_dups;) {
			int dc1 = al_get(complete_dups, j++), dc2 = al_get(complete_dups, j++);
			if (dc1 == d1 && dc2 == d2) {
				verified = 1;
				break;
			}
		}
		if (!verified) {
			printf("kmeans found false dupe between: %s (#%d) & %s (#%d)\n", dataset[d1], d1, dataset[d2], d2);
		}
	}
	
	// Print results.
	// printf("\nCluster Assignments:\n");
	// for (int i = 0; i < NUM_VECTORS; i++) {
	// 	printf("Point %d (in %d): %s\n", i, labels[i], dataset[i]);
	// }
	
	// Print kmeans clustering assignments.
	fprintf(kmeans_file, "\nPoints By Cluster Assignment:\n");
	for (int cluster = 0; cluster < NUM_CLUSTERS; cluster++) {
		fprintf(kmeans_file, "Cluster %d: ", cluster);
		for (int i = 0; i < NUM_VECTORS; i++) {
			if (labels[i] == cluster) {
				fprintf(kmeans_file, "%s, ", dataset[i]);
			}
		}
		fprintf(kmeans_file, "\n");
	}
	
	// Print cluster centroids.
	// "my code is self-documenting"
	size_t cur = 0;
	char buf[NUM_CLUSTERS * (16 + NUM_DIMS * 6)];
	cur += (size_t)snprintf(memset(buf, 0, sizeof(buf)), sizeof(buf) - cur, "\nFinal Centroids:\n");
	for (int j = 0; j < NUM_CLUSTERS; j++) {
		cur += (size_t)snprintf(buf + cur, sizeof(buf) - cur, "Cluster %d: (", j);
		for (int dim = 0; dim < NUM_DIMS; dim++) {
			double val = centroids[j][dim];
			if (val > 0) {
				if (val >= 0.0001) cur += (size_t)snprintf(buf + cur, sizeof(buf) - cur, "%.4lf", val);
				else cur += (size_t)snprintf(buf + cur, sizeof(buf) - cur, "%.4lfe-9", val * 1000 * 1000 * 1000);
			}
			cur += (size_t)snprintf(buf + cur, sizeof(buf) - cur, ",");
		}
		#pragma GCC diagnostic push
		#pragma GCC diagnostic ignored "-Wsequence-point"
		cur += (size_t)snprintf(buf + --cur, sizeof(buf) - cur, ")\n");
		#pragma GCC diagnostic pop
	}
	fprintf(kmeans_file, "%s", buf); // Flush
	
	// Free data
	for (int i = 0; i < NUM_VECTORS; i++) {
		free(vectors[i]);
	}
	free(vectors);
	for (int i = 0; i < NUM_CLUSTERS; i++) {
		free(centroids[i]);
	}
	free(centroids);
	free(labels);
	al_free(complete_dups);
	al_free(sliding_dups);
	al_free(kmeans_dups);
	for (int i = 0; i < NUM_VECTORS; i++) {
		free(dataset[i]);
	}
	
	// Close files.
	if (complete_file) fclose(complete_file);
	if (sliding_file) fclose(sliding_file);
	if (kmeans_file) fclose(kmeans_file);
	
	// End program and flush all buffers.
	printf("\nDone!\n");
	check(fflush(stdout), "fflush(stdout)");
	
	return 0;
}
