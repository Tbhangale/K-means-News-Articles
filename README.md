## K-means Clustering on News Articles

1. Download the news articles dataset named “20news-18828.tar.gz”, from an online textual dataset repository: http://qwone.com/%7Ejason/20Newsgroups/. The 20 Newsgroups data set is a collection of approximately 20,000 newsgroup articles, partitioned (nearly) evenly across 20 different newsgroups.

2. Convert them to a term frequency (TF) matrix. (each row is an article, each column is a unique term, and each entry of this TF matrix is term frequency).

3. Implement the K-means algorithm.

4. K-means algorithm computes the distance of a given data point pair. Replace the computation function with Euclidean distance, 1- Cosine similarity, and 1 – Generalized Jarcard similarity.

5. Run K-means clustering with Euclidean, Cosine and Jarcard similarity. (Specify K as the number of categories of your news articles).

6. Compare the SSEs of Euclidean-K-means Cosine-K-means, Jarcard-K-means. Which method is better and why?

7. Compare the accuracies of Euclidean-K-means Cosine-K-means, Jarcard-K-means. First, label each cluster with the article category of the highest votes. Later, compute the accuracy of the Kmeans with respect to the three similarity metrics. Which metric is better and why?

8. Compare the accuracies (not SSE) of K-means before feature selection and after feature selection. Can feature selection help improve the accuracies of classifiers and why?

9. Which of Euclidean-K-means, Cosine-K-means, Jarcard-K-means requires more iterations and
times and why?
