from copy import deepcopy

import igl
import numpy as np
import scipy as sp
import torch
from nilearn.maskers import SurfaceMasker
from nilearn.surface import SurfaceImage
from pykeops.torch import LazyTensor
from sklearn.base import BaseEstimator, TransformerMixin


def get_functional_features(
    img: SurfaceImage,
    masker: SurfaceMasker,
    hemi: str,
    device: str,
) -> torch.Tensor:
    data = masker.transform(img)
    n_vertices = data.shape[1]
    if hemi == "left":
        data = data[:, : n_vertices // 2]
    else:
        data = data[:, n_vertices // 2 :]
    data = torch.tensor(data, dtype=torch.float32, device=device).T
    return data.contiguous()


def get_vertices_faces(
    img: SurfaceImage, hemi: str
) -> tuple[np.ndarray, np.ndarray]:
    mesh = img.mesh.parts[hemi]
    vertices = mesh.coordinates
    faces = mesh.faces
    return vertices, faces


def get_laplacian_features(
    img: SurfaceImage,
    hemi: str,
    n_lb: int,
    device: str,
) -> torch.Tensor:
    vertices, faces = get_vertices_faces(img, hemi)
    laplacian_matrix = igl.cotmatrix(vertices, faces)
    mass_matrix = igl.massmatrix(vertices, faces, igl.MASSMATRIX_TYPE_VORONOI)
    _, eigenvectors = sp.sparse.linalg.eigsh(
        laplacian_matrix, n_lb, mass_matrix, sigma=0, which="LM"
    )
    eigenvectors = torch.tensor(
        eigenvectors, dtype=torch.float32, device=device
    ).contiguous()
    return eigenvectors


def composite_cost(
    source_features: torch.Tensor,
    target_features: torch.Tensor,
    geom: torch.Tensor,
    alpha: float,
) -> LazyTensor:
    f_i = LazyTensor(source_features, axis=0)
    f_j = LazyTensor(target_features, axis=1)
    l_i = LazyTensor(geom, axis=0)
    l_j = LazyTensor(geom, axis=1)
    cost_functional = ((f_i - f_j) ** 2).sum(-1)
    cost_geometric = ((l_i - l_j) ** 2).sum(-1)
    return alpha * cost_functional + (1 - alpha) * cost_geometric


def sinkhorn_loop(
    C_ij: LazyTensor,
    a_i: torch.Tensor,
    b_j: torch.Tensor,
    eps: float = 1e-3,
    nits: int = 100,
    verbose: bool = False,
) -> LazyTensor:  # -> tuple[ComplexLazyTensor | LazyTensor | Any, ComplexLazyTe...:# -> tuple[ComplexLazyTensor | LazyTensor | Any, ComplexLazyTe...:
    """Straightforward implementation of the Sinkhorn-IPFP-SoftAssign loop in the log domain."""

    # Compute the logarithm of the weights (needed in the softmin reduction) ---
    loga_i, logb_j = a_i.log(), b_j.log()
    loga_i, logb_j = loga_i[:, None, None], logb_j[None, :, None]

    # Setup the dual variables
    U_i, V_j = (
        torch.zeros_like(loga_i),
        torch.zeros_like(logb_j),
    )  # (scaled) dual vectors

    # Sinkhorn loop = coordinate ascent on the dual maximization problem -------
    for i in range(nits):
        if verbose:
            print(f"Iteration {i + 1}/{nits}")
        U_i = -(-C_ij / eps + (V_j + logb_j)).logsumexp(dim=1)[:, None, :]
        V_j = -(-C_ij / eps + (U_i + loga_i)).logsumexp(dim=0)[None, :, :]

    # Optimal plan
    P_ij = (-C_ij / eps + U_i + V_j).exp()

    return P_ij


def fit_one_hemi(
    source_image: SurfaceImage,
    target_image: SurfaceImage,
    hemi: str,
    masker: SurfaceMasker,
    n_lb: int,
    n_iter: int,
    alpha: float,
    reg: float,
    device: str,
    verbose: bool,
) -> LazyTensor:
    source_features = get_functional_features(
        source_image, masker, hemi, device
    )
    target_features = get_functional_features(
        target_image, masker, hemi, device
    )

    geom = get_laplacian_features(source_image, hemi, n_lb, device)

    lazy_cost = composite_cost(
        source_features,
        target_features,
        geom,
        alpha,
    )

    n_vertices = geom.shape[0]
    weights = (
        torch.ones(n_vertices, dtype=torch.float32, device=device) / n_vertices
    )

    lazy_plan = sinkhorn_loop(
        lazy_cost, weights, weights, eps=reg, nits=n_iter, verbose=verbose
    )

    return lazy_plan


def transform_hemi(
    img: SurfaceImage,
    masker: SurfaceMasker,
    lazy_plan: LazyTensor,
    hemi: str,
    device: str,
) -> SurfaceImage:
    data = masker.transform(img)
    source_features = get_functional_features(img, masker, hemi, device)
    projected_features = (
        lazy_plan.T @ source_features / source_features.shape[0]
    )
    data = projected_features.cpu().numpy()
    return data


class SurfaceAlignment(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        masker: SurfaceMasker,
        n_lb: int = 100,
        n_iter: int = 100,
        alpha: float = 0.5,
        reg: float = 1e-3,
        device: str = "cpu",
        verbose: bool = False,
    ) -> None:
        self.masker = masker
        self.n_lb = n_lb
        self.n_iter = n_iter
        self.alpha = alpha
        self.reg = reg
        self.device = device
        self.verbose = verbose

    def fit(
        self, source_image: SurfaceImage, target_image: SurfaceImage
    ) -> None:
        dict_plans = {}
        for hemi in ["left", "right"]:
            plan_hemi = fit_one_hemi(
                source_image,
                target_image,
                hemi,
                self.masker,
                self.n_lb,
                self.n_iter,
                self.alpha,
                self.reg,
                self.device,
                self.verbose,
            )
            dict_plans[hemi] = plan_hemi
        self.dict_plans = dict_plans

    def transform(self, img: SurfaceImage) -> SurfaceImage:
        transformed_img = deepcopy(img)
        for hemi in ["left", "right"]:
            transformed_img.data.parts[hemi] = transform_hemi(
                img,
                self.masker,
                self.dict_plans[hemi],
                hemi,
                self.device,
            )

        return transformed_img
