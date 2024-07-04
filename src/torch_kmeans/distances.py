import torch


def pnorm(x: torch.Tensor, y: torch.Tensor, p: float) -> torch.Tensor:
    """
    Compute the p-norm distance between two tensors.

    Parameters
    ----------
    x : torch.Tensor
        The first tensor.
    y : torch.Tensor
        The second tensor.
    p : float
        The p-norm to compute.

    Returns
    -------
    torch.Tensor
        The p-norm distance between the two tensors.
    """
    return torch.cdist(x, y, p=p)


def euclidean(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Compute the Euclidean distance between two tensors.

    Parameters
    ----------
    x : torch.Tensor
        The first tensor.
    y : torch.Tensor
        The second tensor.

    Returns
    -------
    torch.Tensor
        The Euclidean distance between the two tensors.
    """
    return torch.cdist(x, y, p=2.0)


def cosine(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Compute the cosine distance between two tensors.

    Parameters
    ----------
    x : torch.Tensor
        The first tensor.
    y : torch.Tensor
        The second tensor.

    Returns
    -------
    torch.Tensor
        The cosine distance between the two tensors.
    """
    # Normalize the vectors
    x = torch.nn.functional.normalize(x, p=2, dim=-1)
    y = torch.nn.functional.normalize(y, p=2, dim=-1)

    return 1 - torch.nn.functional.cosine_similarity(x, y, dim=-1)


distances = {
    "euclidean": euclidean,
    "cosine": cosine,
    # "pnorm": pnorm,
}
