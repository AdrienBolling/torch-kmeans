import torch


def silhouette_score(x, labels):
    """
    Compute the silhouette score for each sample using PyTorch with minimal loops.

    Parameters:
    x (torch.Tensor): A tensor of shape (n_samples, n_features) containing the data points.
    labels (torch.Tensor): A tensor of shape (n_samples,) containing the cluster labels for each data point.

    Returns:
    torch.Tensor: A tensor containing the silhouette scores for each sample.
    """

    n_samples = x.size(0)
    unique_labels = labels.unique()

    # Compute pairwise distance matrix
    dist_matrix = torch.cdist(x, x)

    # Initialize A and B tensors
    A = torch.zeros(n_samples, dtype=x.dtype, device=x.device)
    B = torch.full((n_samples,), float('inf'), dtype=x.dtype, device=x.device)

    for label in unique_labels:
        # Create masks for the current cluster and other clusters
        mask = (labels == label)
        other_mask = (labels != label)

        # Compute a(i) for the current cluster
        cluster_distances = dist_matrix[mask][:, mask]
        A[mask] = cluster_distances.sum(dim=1) / (mask.sum() - 1).float()

        # Compute b(i) for the current cluster against all other clusters
        inter_cluster_distances = dist_matrix[mask][:, other_mask].mean(dim=1)
        B[mask] = torch.min(B[mask], inter_cluster_distances)

    # Compute the silhouette scores
    silhouette_scores = (B - A) / torch.max(A, B)

    return silhouette_scores


def average_silhouette_score(x, labels):
    """
    Compute the average silhouette score for all samples using PyTorch.

    Parameters:
    x (torch.Tensor): A tensor of shape (n_samples, n_features) containing the data points.
    labels (torch.Tensor): A tensor of shape (n_samples,) containing the cluster labels for each data point.

    Returns:
    torch.Tensor: The average silhouette score.
    """

    silhouette_scores = silhouette_score(x, labels)

    return silhouette_scores.mean()
