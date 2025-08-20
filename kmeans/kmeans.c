#include <ctype.h>
#include <float.h>
#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define DATASET_PATH "dataset.txt" // Path to the dataset.
#define MAX_ITER 16                // Maximum iterations.
#define NUM_CLUSTERS 64            // Number of clusters (K).
#define NUM_DIMS 251               // The number of dimentions used for clustering data.
#define NUM_VECTORS 1975           // The number of strings in the test dataset.
#define SEED 0                     // The seed used for randomly selecting centroids.

char* dataset[NUM_VECTORS];

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
static int exp_fn_i_dot_product(double* dot_product_ptr, double* r_freq_table1, double* r_freq_table2) {
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
static int exp_fn_i_magnitude(double* magnitude_ptr, double* r_freq_table) {
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
		// table[index] += ((int)temp1 + (int)temp2) % 13 + 1;
		table[index] += 1;
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
		char errorBuffer[BUFSIZ];
		snprintf(errorBuffer, sizeof(errorBuffer), "kmeans.c: Fail - %s", functionName);
		perror(errorBuffer);
		
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
 *** vectors in n dimentional space (where n is NUM_DIMS). In other words,
 *** `similarity(A, B)` performs the function `cos(θ) = (A. B)/(|A| |B|)`
 *** where A and B are vectors.
 ***
 *** Assumes all dimentions of both parameters are assumed to be strictly positive.
 ***
 *** @param v1 The first vector being compared.
 *** @param v2 The second vector being compared.
 *** @returns {0 - 1} where 0 indicates that the vectors (and the strings they
 ***          represent) have no similarity and 1 indicates that they are identical.
 ***/
static double similarity(double* v1, double* v2) {
	// Calculate dot product
	double dot_product = 0;
	check(exp_fn_i_dot_product(&dot_product, v1, v2), "exp_fn_i_dot_product");
	
	// Calculate magnitudes of each relative frequency vector
	double magnitude1 = 0;
	double magnitude2 = 0;
	check(exp_fn_i_magnitude(&magnitude1, v1), "exp_fn_i_magnitude");
	check(exp_fn_i_magnitude(&magnitude2, v2), "exp_fn_i_magnitude");

	// Calculate the similarity score
	return dot_product / (magnitude1 * magnitude2);
}

/*** Inverts the cosine similarity such that more similar vectors produce a
 *** smaller value instead of a larger value.
 ***
 *** Assumes all dimentions of both parameters are assumed to be strictly positive.
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
static double average_cluster_size(double** vectors, int num_vectors, int* labels, double** centroids) {
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
	double overall_avg = 0.0;
	int valid_clusters = 0;
	for (int label = 0; label < NUM_CLUSTERS; label++) {
		int cluster_count = cluster_counts[label];
		if (cluster_count > 0) {
			double cluster_size = cluster_sums[label] / cluster_count;
			double noncluster_size = noncluster_sums[label] / (num_vectors - cluster_count);
			overall_avg += cluster_size;
			valid_clusters++;
			
			printf(
				"> Cluster #%d (x%d): %.4lf (vs. %.4lf).\n",
				label, cluster_count, cluster_size, noncluster_size
			); // Debug
		}
	}
	
	// If no clusters have any points, return 0.
	return (valid_clusters > 0) ? (overall_avg / valid_clusters) : 0.0;
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
		printf("Centroid %d starts at vector %d.\n", i, random_index); // Debug
		
		// Copy each dimetion from the selected random vector to the current centroid.
		for (int dim = 0; dim < NUM_DIMS; dim++) {
			centroids[i][dim] = vectors[random_index][dim];
		}
		
		print_difference(vectors, 0, random_index); // Debug
	}
	printf("\n"); // Debug
	
	// Allocate memory for new centroids
	double** new_centroids = (double**) malloc(NUM_CLUSTERS * sizeof(double*));
	for (int i = 0; i < NUM_CLUSTERS; i++) {
		new_centroids[i] = create_vector();
	}
	
	// Main loop
	int* cluster_sizes = (int*) malloc(NUM_CLUSTERS * sizeof(int));
	for (int i = 0; i < MAX_ITER; i++) {
		bool changed = false;
		
		// Reset new centroids
		for (int i = 0; i < NUM_CLUSTERS; i++) {
			for (int dim = 0; dim < NUM_DIMS; dim++) {
				new_centroids[i][dim] = 0.0;
			}
			cluster_sizes[i] = 0;
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
			cluster_sizes[best_centroid_label]++;
		}
		
		// Update centroids
		for (int j = 0; j < NUM_CLUSTERS; j++) {
			if (cluster_sizes[j] > 0) {
				for (int dim = 0; dim < NUM_DIMS; dim++) {
					centroids[j][dim] /= cluster_sizes[j];
				}
			}
		}
		
		printf("Average cluster size is %.6lf.\n\n", average_cluster_size(vectors, num_vectors, labels, centroids));
		check(fflush(stdout), "fflush");
		
		// Stop if centroids didn't change.
		if (!changed) break;
	}
	
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
}

// Not thread safe.
static void load_dataset(void) {
	FILE *file = fopen(DATASET_PATH, "r");
	if (!file) {
		char errorBuffer[BUFSIZ];
		snprintf(errorBuffer, sizeof(errorBuffer), "Failed to open file: %s", DATASET_PATH);
		perror(errorBuffer);
		exit(EXIT_FAILURE);
	}
	
	char* buffer = NULL;
	size_t bufsize = 0;
	int error = getdelim(&buffer, &bufsize, EOF, file) == -1;
	fclose(file);
	
	if (error) {
		char errorBuffer[BUFSIZ];
		snprintf(errorBuffer, sizeof(errorBuffer), "Failed to read file: %s", DATASET_PATH);
		perror(errorBuffer);
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

int main(void) {
	// Set stdout to only flush manually with a 8MB buffer.
	setvbuf(stdout, NULL, _IOFBF, (8 * 1000 * 1000));
	
	// Load the dataset from the .gitignored dataset file.
	load_dataset();
	
	// Allocate ram to store vectors.
	double** vectors = malloc(NUM_VECTORS * sizeof(double*));
	
	// Build the vectors.
	check(build_vectors(vectors, dataset, NUM_VECTORS), "build_vectors");
	
	// Debug printing to test that difference is working.
	print_difference(vectors, 0, 1);
	printf("\n");
	
	// Allocate memory for clustering.
	int* labels = malloc(NUM_VECTORS * sizeof(int));
	double** centroids = malloc(NUM_CLUSTERS * sizeof(double*));
	for (int i = 0; i < NUM_CLUSTERS; i++) {
		centroids[i] = malloc(NUM_DIMS * sizeof(double));
	}
	
	// Run the clustering algorithm.
	kmeans(vectors, NUM_VECTORS, labels, centroids);
	
	// Print results.
	printf("\nCluster Assignments:\n");
	for (int i = 0; i < NUM_VECTORS; i++) {
		printf("Point %d (in %d): %s\n", i, labels[i], dataset[i]);
	}
	
	// Print results grouped by cluster.
	printf("\nPoints By Cluster Assignment:\n");
	for (int cluster = 0; cluster < NUM_CLUSTERS; cluster++) {
		printf("Cluster %d: ", cluster);
		for (int i = 0; i < NUM_VECTORS; i++) {
			if (labels[i] == cluster) {
				printf("%s,", dataset[i]);
			}
		}
		printf("\n");
	}
	
	printf("\nFinal Centroids:\n");
	for (int j = 0; j < NUM_CLUSTERS; j++) {
		printf("Cluster %d: (", j);
		for (int dim = 0; dim < NUM_DIMS; dim++) {
			printf("%.8lf", centroids[j][dim]);
			if (dim < NUM_DIMS - 1) printf(", ");
		}
		printf(")\n");
	}
	
	printf("Done!\n");
	check(fflush(stdout), "fflush");
	
	return 0;
}
