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

// Define the dataset.
#define DATASET_PATH "datasets/dataset_unique_sorted.txt"
#define DATASET_SIZE 1257 // 1975

#define NUM_DIMS 251               // The number of dimensions used for clustering data.
#define NUM_VECTORS DATASET_SIZE   // The number of strings in the test dataset.
#define SEED 1621963727            // The seed used for randomly selecting centroids.
#define THRESHOLD 0.75

char* dataset[NUM_VECTORS];
FILE* complete_file = NULL;
FILE* sliding_file = NULL;
FILE* kmeans_file = NULL;

#define streql(str1, str2) (strcmp((str1), (str2)) == 0)

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
int exp_fn_i_hash_char_pair(double num1, double num2) {
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
int exp_fn_i_frequency_table(double* table, char* term) {
	// Initialize hash table with 0 values
	for (int i = 0; i < NUM_DIMS; i++) table[i] = 0;
	
	// j is the former character, i is the latter.
	int num_terms = (int)strlen(term);
	for (int j = -1, i = 0; i <= num_terms; i++) {
		// If latter character is punctuation or whitespace, skip it.
		if (ispunct(term[i]) || isspace(term[i])) continue;

		int temp1 = (j == -1) ? '`' : (int)tolower(term[j]);
		int temp2 = (i == num_terms) ? '`' : (int)tolower(term[i]);

		// If either character is a number, reassign the code point
		// The significance of 75 here seems to be that it puts the numbers
		// right after the end of the lowercase letters, but they will still
		// colide with the {, |, }, ~ and DEL ASCII characters.
		if ('0' <= temp1 && temp1 <= '9') temp1 += 75;
		if ('0' <= temp2 && temp2 <= '9') temp2 += 75;
		
		// Hash the character pair into an index.
		int index = exp_fn_i_hash_char_pair(temp1, temp2);
		
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
int build_vectors(double** vectors, char** strs, size_t num_vectors) {
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
double similarity(const double* v1, const double* v2) {
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
int print_cluster_size(double** vectors, int* labels, double** centroids, int num_clusters, int iteration) {
	double cluster_sums[num_clusters];
	double noncluster_sums[num_clusters];
	int cluster_counts[num_clusters];
	memset(cluster_sums, 0, sizeof(cluster_sums));
	memset(noncluster_sums, 0, sizeof(noncluster_sums));
	memset(cluster_counts, 0, sizeof(cluster_counts));
	
	// Sum the difference from each vector to its cluster centroid.
	for (int i = 0; i < NUM_VECTORS; i++) {
		int label = labels[i];
		double* vector = vectors[i];
		cluster_sums[label] += difference(vector, centroids[label]);
		cluster_counts[label]++;
		
		for (int j = 0; j < num_clusters; j++) {
			if (j == label) continue;
			noncluster_sums[j] += difference(vector, centroids[j]);
		}
	}
	
	// Calculate the average difference per cluster and then the overall average.
	fprintf(kmeans_file, "Cluster Sizes:\n");
	int valid_clusters = 0;
	double cluster_total = 0.0, noncluster_total = 0.0;
	double max_cluster_size = 0.0, min_cluster_size = 1.0;
	int max_cluster_label = -1, min_cluster_label = -1;
	for (int label = 0; label < num_clusters; label++) {
		int cluster_count = cluster_counts[label];
		if (cluster_count > 0) {
			double cluster_size = cluster_sums[label] / cluster_count;
			double noncluster_size = noncluster_sums[label] / (NUM_VECTORS - cluster_count);
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
		
		fprintf(kmeans_file,
			"\nkmeans #%d:\n"
				"\t> Average cluster: %.4lf\n"
				"\t> Average noncluster: %.4lf\n"
				"\t> Largest cluster: #%d @ %.4lf\n"
				"\t> Smallest cluster: #%d @ %.4lf\n"
			"\n",
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
void kmeans(double** vectors, int* labels, double** centroids, int max_iter, int num_clusters) {
	// Select random vectors to use as the initial centroids.
	srand(SEED);
	for (int i = 0; i < num_clusters; i++) {
		// Pick a random vector.
		const int random_index = rand() % NUM_VECTORS;
		fprintf(kmeans_file, "Centroid %d starts at vector %d.\n", i, random_index); // Debug
		
		// Copy each dimetion from the selected random vector to the current centroid.
		for (int dim = 0; dim < NUM_DIMS; dim++) {
			centroids[i][dim] = vectors[random_index][dim];
		}
		
		// print_difference(vectors, 0, random_index); // Debug
	}
	fprintf(kmeans_file, "\n");
	
	// Allocate memory for new centroids
	double** new_centroids = malloc((size_t)num_clusters * sizeof(double*));
	for (int i = 0; i < num_clusters; i++) {
		new_centroids[i] = create_vector();
	}
	
	// Main loop
	int* cluster_counts = malloc((size_t)num_clusters * sizeof(int));
	for (int i = 0; i < max_iter; i++) {
		bool changed = false;
		
		// Reset new centroids
		for (int i = 0; i < num_clusters; i++) {
			for (int dim = 0; dim < NUM_DIMS; dim++) {
				new_centroids[i][dim] = 0.0;
			}
			cluster_counts[i] = 0;
		}
		
		// Assign each point to the nearest centroid
		for (int i = 0; i < NUM_VECTORS; i++) {
			double* vector = vectors[i];
			double min_dist = DBL_MAX;
			int best_centroid_label = 0;
			
			for (int j = 0; j < num_clusters; j++) {
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
		for (int j = 0; j < num_clusters; j++) {
			if (cluster_counts[j] > 0) {
				for (int dim = 0; dim < NUM_DIMS; dim++) {
					centroids[j][dim] = new_centroids[j][dim] / cluster_counts[j];
				}
			}
		}
		
		print_cluster_size(vectors, labels, centroids, num_clusters, i);
		
		// Stop if centroids didn't change.
		if (!changed) break;
	}
	
	// Final centroid update.
	for (int i = 0; i < NUM_VECTORS; i++) {
		double* vector = vectors[i];
		double min_dist = DBL_MAX;
		int best_centroid_label = 0;
		
		for (int j = 0; j < num_clusters; j++) {
			double dist = difference(vector, centroids[j]);
			if (dist < min_dist) {
				min_dist = dist;
				best_centroid_label = j;
			}
		}
		labels[i] = best_centroid_label;
	}
	
	// Free memory.
	for (int i = 0; i < num_clusters; i++) {
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
void load_dataset(const char* dataset_path) {
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

/*** Checks whether the ordered pair (d1, d2) exists in `complete_dups`.
 ***
 *** @param complete_dups An ArrayList containing pairs of duplicate indices.
 *** @param d1 First index of the pair to verify.
 *** @param d2 Second index of the pair to verify.
 *** @returns true if the exact pair exists, false otherwise.
 ***/
bool verify_dupe(ArrayList* complete_dups, int d1, int d2) {
	size_t num_complete_dups = complete_dups->size;
	for (size_t j = 0; j < num_complete_dups;) {
		int dc1 = al_get(complete_dups, j++);
		int dc2 = al_get(complete_dups, j++);
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
 ***/
ArrayList* find_complete_dups(double** vectors) {
	ArrayList* complete_dups = al_newc(512);
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
	
	// Lock results.
	al_lock(complete_dups);
	al_trim_to_size(complete_dups);
	
	// Log duplocates found by the complete strategy.
	fprintf(complete_file, "Duplocates found:\n");
	size_t num_complete_dups = complete_dups->size;
	for (size_t i = 0; i < num_complete_dups;) {
		int d1 = al_get(complete_dups, i++), d2 = al_get(complete_dups, i++);
		fprintf(complete_file, "%s (#%d) & %s (#%d)\n", dataset[d1], d1, dataset[d2], d2);
	}
	
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
 ***/
ArrayList* find_sliding_dups(double** vectors, int window_size, ArrayList* complete_dups) {
	ArrayList* sliding_dups = al_newc(512);
	for (int i = 0; i < NUM_VECTORS; i++) {
		const double* v1 = vectors[i];
		const int j_max = min(i + window_size, NUM_VECTORS);
		for (int j = i + 1; j < j_max; j++) {
			const double* v2 = vectors[j];
			if (similarity(v1, v2) > THRESHOLD) {
				al_add(sliding_dups, i);
				al_add(sliding_dups, j);
			}
		}
	}
	
	// Lock results.
	al_lock(sliding_dups);
	al_trim_to_size(sliding_dups);
	
	// Log and verify duplocates found by the sliding strategy.
	size_t num_sliding_dups = sliding_dups->size;
	fprintf(sliding_file, "Duplocates found:\n");
	for (size_t i = 0; i < num_sliding_dups;) {
		int d1 = al_get(sliding_dups, i++), d2 = al_get(sliding_dups, i++);
		fprintf(sliding_file, "%s (#%d) & %s (#%d)\n", dataset[d1], d1, dataset[d2], d2);
		
		if (!verify_dupe(complete_dups, d1, d2)) {
			printf("sliding found false dupe: %s (#%d) & %s (#%d)\n", dataset[d1], d1, dataset[d2], d2);
		}
	}
	
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
 ***/
ArrayList* find_kmeans_dups(double** vectors, int max_iter, int num_clusters, ArrayList* complete_dups) {
	// Malloc memory for finding clusters.
	int* labels = malloc(NUM_VECTORS * sizeof(int));
	for (int i = 0; i < NUM_VECTORS; i++) labels[i] = -1;
    double** centroids = malloc((size_t)num_clusters * sizeof(double*));
	for (int i = 0; i < num_clusters; i++)
		centroids[i] = malloc(NUM_DIMS * sizeof(double));
	
	// Execute kmeans clustering.
	kmeans(vectors, labels, centroids, max_iter, num_clusters);
	
	// Find duplocates in clusters.
	ArrayList* kmeans_dups = al_newc(512);
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
	
	// Lock results.
	al_lock(kmeans_dups);
	al_trim_to_size(kmeans_dups);
	
	// Log duplocates found by the kmeans strategy.
	fprintf(kmeans_file, "Duplocates found:\n");
	size_t num_kmeans_dups = kmeans_dups->size;
	for (size_t i = 0; i < num_kmeans_dups;) {
		int d1 = al_get(kmeans_dups, i++), d2 = al_get(kmeans_dups, i++);
		fprintf(kmeans_file, "%s (#%d) & %s (#%d)\n", dataset[d1], d1, dataset[d2], d2);
		
		if (!verify_dupe(complete_dups, d1, d2)) {
			printf("kmeans found false dupe: %s (#%d) & %s (#%d)\n", dataset[d1], d1, dataset[d2], d2);
		}
	}
	
	// Print kmeans clusters.
	fprintf(kmeans_file, "\nPoints By Cluster Assignment:\n");
	for (int cluster = 0; cluster < num_clusters; cluster++) {
		fprintf(kmeans_file, "Cluster %d: ", cluster);
		for (int i = 0; i < NUM_VECTORS; i++) {
			if (labels[i] == cluster) {
				fprintf(kmeans_file, "%s, ", dataset[i]);
			}
		}
		fprintf(kmeans_file, "\n");
	}
	
	// Print cluster centroids. "my code is self-documenting"
	size_t cur = 0;
	char buf[num_clusters * (16 + NUM_DIMS * 6)];
	cur += (size_t)snprintf(memset(buf, 0, sizeof(buf)), sizeof(buf) - cur, "\nFinal Centroids:\n");
	for (int j = 0; j < num_clusters; j++) {
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
	
	// Free memory.
	free(labels);
	for (int i = 0; i < num_clusters; i++) {
		free(centroids[i]);
	}
	free(centroids);
	
	return kmeans_dups;
}

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
	
	// Set buffers to only flush manually for more accurate performance evaluation.
	setvbuf(stdout, NULL, _IOFBF, (2 * 1000 * 1000));
	setvbuf(kmeans_file, NULL, _IOFBF, (4 * 1000 * 1000));
	printf("Begin!\n");
	check(fflush(stdout), "fflush(stdout)");
	
	// Load dataset and build vectors.
	load_dataset(DATASET_PATH);
	double** vectors = malloc(NUM_VECTORS * sizeof(double*));
	check(build_vectors(vectors, dataset, NUM_VECTORS), "build_vectors");
	
	// Execute the complete.
	printf("\n");
	check(fflush(stdout), "fflush(stdout)");
	Timer* timer = timer_new();
	timer_start(timer);
	ArrayList* complete_dups = find_complete_dups(vectors);
	timer_stop(timer);
	timer_store(timer);
	
	// Print complete summary.
	timer_print(timer, "Complete similarity");
	printf("Complete search found %ld dups.\n", complete_dups->size / 2);
	check(fflush(stdout), "fflush(stdout)");
	
	// Execute current (sliding window).
	printf("\n");
	check(fflush(stdout), "fflush(stdout)");
	timer_start(timer);
	ArrayList* sliding_dups = find_sliding_dups(vectors, 6, complete_dups);
	timer_stop(timer);
	
	// Print sliding summary.
	timer_print_cmp(timer, "Sliding similarity");
	double percent_success_sliding = 100.0 * (double)sliding_dups->size / (double)complete_dups->size;
	printf("Sliding window found %ld dups (%%%.2lf).\n", sliding_dups->size / 2, percent_success_sliding);
	check(fflush(stdout), "fflush(stdout)");
	
	// All kmeans use up to 16 iterations.
	int max_iter = 16;
	
	// Execute kmeans clustering.
	printf("\n");
	check(fflush(stdout), "fflush(stdout)");
	timer_start(timer);
	ArrayList* kmeans_dups = find_kmeans_dups(vectors, max_iter, 64, complete_dups);
	timer_stop(timer);
	timer_print_cmp(timer, "Kmeans clustering");
	
	// Print kmeans summary.
	double percent_success_kmeans = 100.0 * (double)kmeans_dups->size / (double)complete_dups->size;
	printf("kmeans found %ld dups (%%%.2lf).\n", kmeans_dups->size / 2, percent_success_kmeans);
	check(fflush(stdout), "fflush(stdout)");
	
	// Free memory.
	for (int i = 0; i < NUM_VECTORS; i++) {
		free(vectors[i]);
	}
	free(vectors);
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
