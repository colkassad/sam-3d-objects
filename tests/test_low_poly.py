import numpy as np
import pyvista as pv
import torch

from sam3d_objects.model.backbone.tdfy_dit.representations.mesh.cube2mesh import (
    MeshExtractResult,
)
from sam3d_objects.model.backbone.tdfy_dit.utils.postprocessing_utils import (
    simplify_mesh_representation,
    to_glb,
)


def make_colored_sphere():
    sphere = pv.Sphere(theta_resolution=24, phi_resolution=24).triangulate()
    vertices = np.asarray(sphere.points).copy()
    faces = np.asarray(sphere.faces).reshape(-1, 4)[:, 1:].copy()
    color = (vertices - vertices.min(axis=0)) / np.ptp(vertices, axis=0)
    attrs = np.concatenate([color, np.zeros_like(color)], axis=1)
    return MeshExtractResult(
        torch.from_numpy(vertices),
        torch.from_numpy(faces),
        vertex_attrs=torch.from_numpy(attrs.astype(np.float16)),
    )


def test_face_budget_preserves_vertex_attributes():
    mesh = make_colored_sphere()

    simplified = simplify_mesh_representation(mesh, target_faces=120)

    assert simplified.faces.shape[0] <= 125
    assert simplified.faces.shape[0] > 0
    assert simplified.vertex_attrs.shape[0] == simplified.vertices.shape[0]
    assert simplified.vertex_attrs.shape[1] == 6
    assert simplified.vertex_attrs.dtype == torch.float16
    assert torch.isfinite(simplified.vertex_attrs).all()
    assert simplified.vertices.device.type == "cpu"


def test_flat_shading_unmerges_vertices_and_preserves_colors():
    mesh = simplify_mesh_representation(make_colored_sphere(), target_faces=120)

    glb = to_glb(
        None,
        mesh,
        with_mesh_postprocess=False,
        with_texture_baking=False,
        use_vertex_color=True,
        flat_shading=True,
    )

    assert len(glb.vertices) == len(glb.faces) * 3
    assert len(glb.visual.vertex_colors) == len(glb.vertices)
