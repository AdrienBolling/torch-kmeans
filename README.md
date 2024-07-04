# torch-kmeans
Pytorch implementation of KMeans


This implementation ahs been inspired in part by [This work](subhadarship.github.io/kmeans_pytorch) . However this repo has not been updated in a while and I wanted to add some additional features, mainly a class implementation as I find it easier to handle, as well as some metrics.

## Installation
```bash
pip install torch-kmeans
```

## Usage

```python
from src.torch_kmeans import KMeans

kmeans = KMeans(max_iter=100, eps=1e-4, distance='euclidean')
```

## Parameters

- `max_iter`: int, default=100
    Maximum number of iterations of the k-means algorithm for a single run.
- `eps`: float, default=1e-4
    Relative tolerance in regard to inertia to declare convergence.
- `distance`: str, default='euclidean'
    Distance metric to use. Can be 'euclidean' or 'cosine'

## Attributes

- `centroids`: tensor of shape (n_clusters, n_features)
    Coordinates of cluster centers.
- `labels`: tensor of shape (n_samples,)
    Labels of each point
- `inertia`: float
    Sum of squared distances of samples to their closest cluster center.
- `n_clusters`: int
    Number of iterations run.

## Methods

- `fit(x, n_clusters: Union[int, str] = 'auto', sweep: Union[Iterable[int], str] = 'fast', verbose: bool = False)`: 
    Fit the KMeans model to the data. If `n_clusters` is 'auto', the algorithm will sweep through a range of values and select the one with the lowest inertia. If `sweep` is 'fast', the algorithm will only sweep through a few values. If `verbose` is True, the algorithm will print the progress of the sweep.
- `predict(x)`: 
    Predict the closest cluster each sample in `x` belongs to.
- `fit_predict(x, n_clusters: Union[int, str] = 'auto', sweep: Union[Iterable[int], str] = 'fast', verbose: bool = False)`: 
    Fit the KMeans model to the data and return the labels.
- `silhouette_score(x)`: 
    Compute the silhouette score of the clustering.
- `average_silhouette_score(x)`: 
    Compute the average silhouette score of the clustering.


## Metrics
In the `torch_kmeans.metrics` module, you can find these metrics implemented:
- `silhouette_score(x)`: 
    Compute the silhouette score of the clustering. The silhouette score is a measure of how similar an object is to its own cluster (cohesion) compared to other clusters (separation). The silhouette score ranges from -1 to 1, where a high value indicates that the object is well matched to its own cluster and poorly matched to neighboring clusters.
- `average_silhouette_score(x)`: 
    Compute the average silhouette score of the clustering. The average silhouette score is the mean of the silhouette scores of all samples.

## Distances
In the `torch_kmeans.distances` module, you can find these distances implemented:
- `euclidean(x, y)`: 
    Compute the Euclidean distance between two tensors.
- `cosine(x, y)`: 
    Compute the cosine distance between two tensors.

## Example

### Basic usage

```python
import torch
from src.torch_kmeans import KMeans

x = torch.rand(100, 2)
kmeans = KMeans(max_iter=100, eps=1e-4, distance='euclidean')
kmeans.fit(x, n_clusters=3)
print(kmeans.labels)
```

### Auto selection of number of clusters

```python
import torch
from src.torch_kmeans import KMeans

x = torch.rand(100, 2)
kmeans = KMeans(max_iter=100, eps=1e-4, distance='euclidean')
kmeans.fit(x, n_clusters='auto')
print(kmeans.labels)
```

### Silhouette score

```python
import torch
from src.torch_kmeans import KMeans

x = torch.rand(100, 2)
kmeans = KMeans(max_iter=100, eps=1e-4, distance='euclidean')
kmeans.fit(x, n_clusters=3)
print(kmeans.silhouette_score(x))
```