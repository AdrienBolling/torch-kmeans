import torch
from typing import Union
from collections.abc import Iterable

from metrics import silhouette_score
from distances import distances

from tqdm import tqdm


class KMeans:

    def __init__(
            self,
            max_iter: int = 100,
            num_init: int = 10,
            eps: float = 1e-4,
            distance: str = "euclidean",
    ):
        """
        Initialize the KMeans object.

        Parameters
        ----------
        max_iter : int
            The maximum number of iterations.
        num_init : int
            The number of initializations to try.
        eps : float
            The convergence threshold.
        distance : str
            The distance metric to use. Currently only supports "euclidean" and "cosine".
        """
        # Check arguments are valid
        if distance not in distances:
            raise ValueError(f"distance must be one of {list(distances.keys())}")

        if max_iter <= 0:
            raise ValueError("max_iter must be positive")

        if num_init <= 0:
            raise ValueError("num_init must be positive")

        if eps <= 0:
            raise ValueError("eps must be positive")

        # Set hyperparameters
        self.max_iter = max_iter
        self.num_init = num_init
        self.eps = eps
        self.distance = distances[distance]

        # Declaration of other attributes
        self.centroids = None
        self.labels = None
        self.inertia = None
        self.n_clusters = None

    def _fit_with_fixed_n_clusters(self, x: torch.Tensor, n_clusters: int, verbose: bool = False):
        """
        Fit the KMeans model to the data with a fixed number of clusters.

        Parameters
        ----------
        x : torch.Tensor
            The data to cluster.
        n_clusters : int
            The number of clusters to form as well as the number of centroids to generate.
        """
        labels = None
        distance = None

        # Over multiple initializations
        for _ in tqdm(range(self.num_init), desc="Trying Initialization", disable=not verbose):

            # Initialize centroids
            self.centroids = x[torch.randperm(x.size(0))[:n_clusters]]

            # Main loop
            for _ in tqdm(range(self.max_iter), desc="Fitting KMeans", disable=not verbose):
                # Compute distances
                distance = self.distance(x, self.centroids)

                # Assign clusters
                labels = torch.argmin(distance, dim=-1)

                # Update centroids
                new_centroids = torch.stack([x[labels == i].mean(dim=0) for i in range(n_clusters)])

                # Check convergence
                if torch.all(torch.norm(new_centroids - self.centroids, dim=-1) < self.eps):
                    break

                self.centroids = new_centroids

        # Compute inertia
        self.inertia = torch.sum(torch.min(distance, dim=-1).values)

        self.labels = labels

    def fit(self,
            x: torch.Tensor,
            n_clusters: Union[int, str] = 'auto',
            sweep: Union[Iterable[int], str] = 'fast',
            verbose: bool = False
            ):
        """
        Fit the KMeans model to the data.

        Parameters
        ----------
        x : torch.Tensor
            The data to cluster.
        n_clusters : int or str
            The number of clusters to form as well as the number of centroids to generate.
            If 'auto', the optimal number of clusters is determined using the Average Silhouette method.
        sweep : Iterable[int] or str
            The number of clusters to try. If 'fast', a small number of clusters are tried, from 1 to log(number of
            points) + 1. Ignored if n_clusters is not 'auto'.
        verbose : bool
            Whether to print progress information.
        """
        # Check arguments are valid
        if not isinstance(x, torch.Tensor):
            raise ValueError("x must be a torch.Tensor")
        if len(x.size()) != 2:
            raise ValueError("x must be a 2D tensor")
        if x.size(0) < 2:
            raise ValueError("x must have at least 2 samples")

        if n_clusters != 'auto' and not isinstance(n_clusters, int):
            raise ValueError("n_clusters must be an integer or 'auto'")

        if sweep != 'fast' and not isinstance(sweep, Iterable):
            raise ValueError("sweep must be an iterable or 'fast'")

        if not isinstance(verbose, bool):
            raise ValueError("verbose must be a boolean")

        if n_clusters == 'auto':

            if sweep == 'fast':
                ceiling = torch.log(torch.tensor(x.size(0))).ceil().int().item()
                sweep = range(1, ceiling)

            elif isinstance(sweep, Iterable):
                pass

            best_score = torch.FloatTensor([1])
            best_n_clusters = None

            for n_clusters in tqdm(sweep, disable=not verbose, desc="Finding optimal number of clusters"):
                self._fit_with_fixed_n_clusters(x, n_clusters)
                score = silhouette_score(x, self.labels).mean().item()

                if torch.abs(score) < best_score:
                    best_score = score
                    best_n_clusters = n_clusters
            if verbose:
                print(f"Optimal number of clusters: {best_n_clusters} with silhouette score {best_score}")

        else:
            best_n_clusters = n_clusters

        self._fit_with_fixed_n_clusters(x, best_n_clusters)
        self.n_clusters = best_n_clusters

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Predict the closest cluster each sample in x belongs to.

        Parameters
        ----------
        x : torch.Tensor
            The data to predict.

        Returns
        -------
        torch.Tensor
            The predicted cluster for each sample.
        """

        # Check arguments are valid
        if not isinstance(x, torch.Tensor):
            raise ValueError("x must be a torch.Tensor")
        if x.size(-1) != self.centroids.size(-1):
            raise ValueError("x must have the same number of features as the centroids")
        if len(x.size()) != 2:
            raise ValueError("x must be a 2D tensor")
        if x.size(0) < 2:
            raise ValueError("x must have at least 2 samples")

        # Check if fit has been called
        if self.centroids is None:
            raise ValueError("fit must be called before predict")

        distance = self.distance(x, self.centroids)
        return torch.argmin(distance, dim=-1)

    def fit_predict(self, x: torch.Tensor, n_clusters: Union[int, str] = 'auto',
                    sweep: Union[Iterable[int], str] = 'fast', verbose: bool = False) -> torch.Tensor:
        """
        Fit the KMeans model to the data and predict the closest cluster each sample belongs to.

        Parameters
        ----------
        x : torch.Tensor
            The data to cluster.
        n_clusters : int or str
            The number of clusters to form as well as the number of centroids to generate.
            If 'auto', the optimal number of clusters is determined using the Average Silhouette method.
        sweep : Iterable[int] or str
            The number of clusters to try. If 'fast', a small number of clusters are tried, from 1 to log(number of
            points) + 1.
        verbose : bool
            Whether to print progress information.

        Returns
        -------
        torch.Tensor
            The predicted cluster for each sample.
        """

        # Fit the model
        self.fit(x, n_clusters, sweep, verbose)

        # Return the predicted labels
        return self.labels

    def silhouette_score(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute the silhouette score for each sample using PyTorch with minimal loops.

        Parameters:
        x (torch.Tensor): A tensor of shape (n_samples, n_features) containing the data points.

        Returns:
        torch.Tensor: A tensor containing the silhouette scores for each sample.
        """

        return silhouette_score(x, self.labels)

    def average_silhouette_score(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute the average silhouette score for all samples using PyTorch.

        Parameters:
        x (torch.Tensor): A tensor of shape (n_samples, n_features) containing the data points.

        Returns:
        torch.Tensor: The average silhouette score.
        """

        return self.silhouette_score(x).mean()

    def __repr__(self):
        return (f"KMeans(n_clusters={self.n_clusters}, max_iter={self.max_iter}, num_init={self.num_init}, "
                f"eps={self.eps}, distance={self.distance.__name__})")

    def __str__(self):
        return self.__repr__()

    def __call__(self, x: torch.Tensor, n_clusters: Union[int, str] = 'auto', sweep: Union[Iterable[int], str] = 'fast',
                 verbose: bool = False) -> torch.Tensor:
        return self.fit_predict(x, n_clusters, sweep, verbose)
