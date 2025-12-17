import numpy as np
from typing import Dict


def get_fisheye_metadata(
    bb_xywh: np.ndarray,
    image_width: int,
    image_height: int,
    focal_length: float,
) -> Dict[str, float]:
    """
    Computes geometric and optical metadata for an object in a fisheye scene.

    This function uses the Equidistant (f-theta) projection model to derive
    spatial features from standard COCO-style bounding boxes. These values help
    the model understand radial distortion and the non-linear pixel density
    characteristic of wide-angle lenses.

    Args:
        bb (np.ndarray): Bounding box in [x_min, y_min, width, height] format.
        image_width (int): The width of the image in pixels.
        image_height (int): The height of the image in pixels.
        focal_length (float): The pixel focal length (f) of the camera.
            Typically calculated as R / (max_FOV / 2).

    Returns:
        dict: A dictionary containing the following computed metadata:
            - "radial_dist": Normalized distance [0.0, 1.0] from the image
              center to the object. Higher values indicate objects near the
              distorted periphery.
            - "phi": The azimuthal angle (angular position) of the object
              relative to the center, ranging from (-pi, pi]. Useful for
              understanding rotation-dependent appearance.
            - "theta": The incident angle (polar angle) in radians between
              the light ray and the optical axis. Derived via the
              equidistant model r = f * theta.
            - "distortion_factor": A proxy for local pixel compression.
              Values closer to 1.0 at the edges indicate higher "label noise"
              where rectangular boxes poorly fit curved objects.
    """
    cx, cy = image_width / 2.0, image_height / 2.0
    u = float(bb_xywh[0] + bb_xywh[2] / 2.0)
    v = float(bb_xywh[1] + bb_xywh[3] / 2.0)

    max_r = float(np.sqrt(cx**2 + cy**2))
    r = float(np.sqrt((u - cx) ** 2 + (v - cy) ** 2))
    radial_dist = r / max_r if max_r > 0 else 0.0
    phi = float(np.arctan2(v - cy, u - cx))
    theta = float(r / focal_length) if focal_length > 0 else 0.0
    distortion_factor = float(np.sin(theta)) if theta < (np.pi / 2.0) else 1.0

    return {
        "radial_dist": radial_dist,
        "phi": phi,
        "theta": theta,
        "distortion_factor": distortion_factor,
    }


def fisheye_stats(
    boxes_xyxy: np.ndarray,
    *,
    orig_w: int,
    orig_h: int,
    focal_length: float,
    nan_default: float,
) -> Dict[str, float]:
    """Aggregate fisheye metadata across all bboxes in a sample."""
    if boxes_xyxy.size == 0:
        return {
            "fisheye_radial_dist_mean": nan_default,
            "fisheye_radial_dist_median": nan_default,
            "fisheye_radial_dist_min": nan_default,
            "fisheye_radial_dist_max": nan_default,
            "fisheye_phi_mean": nan_default,
            "fisheye_phi_median": nan_default,
            "fisheye_phi_min": nan_default,
            "fisheye_phi_max": nan_default,
            "fisheye_theta_mean": nan_default,
            "fisheye_theta_median": nan_default,
            "fisheye_theta_min": nan_default,
            "fisheye_theta_max": nan_default,
            "fisheye_distortion_mean": nan_default,
            "fisheye_distortion_median": nan_default,
            "fisheye_distortion_min": nan_default,
            "fisheye_distortion_max": nan_default,
        }

    xywh = boxes_xyxy.copy()
    xywh[:, 2] = xywh[:, 2] - xywh[:, 0]
    xywh[:, 3] = xywh[:, 3] - xywh[:, 1]

    radial = []
    phi = []
    theta = []
    distortion = []
    for bb in xywh:
        meta = get_fisheye_metadata(bb, orig_w, orig_h, focal_length)
        radial.append(meta["radial_dist"])
        phi.append(meta["phi"])
        theta.append(meta["theta"])
        distortion.append(meta["distortion_factor"])

    radial = np.asarray(radial, dtype=np.float32)
    phi = np.asarray(phi, dtype=np.float32)
    theta = np.asarray(theta, dtype=np.float32)
    distortion = np.asarray(distortion, dtype=np.float32)

    return {
        "fisheye_radial_dist_mean": float(radial.mean()),
        "fisheye_radial_dist_median": float(np.median(radial)),
        "fisheye_radial_dist_min": float(radial.min()),
        "fisheye_radial_dist_max": float(radial.max()),
        "fisheye_phi_mean": float(phi.mean()),
        "fisheye_phi_median": float(np.median(phi)),
        "fisheye_phi_min": float(phi.min()),
        "fisheye_phi_max": float(phi.max()),
        "fisheye_theta_mean": float(theta.mean()),
        "fisheye_theta_median": float(np.median(theta)),
        "fisheye_theta_min": float(theta.min()),
        "fisheye_theta_max": float(theta.max()),
        "fisheye_distortion_mean": float(distortion.mean()),
        "fisheye_distortion_median": float(np.median(distortion)),
        "fisheye_distortion_min": float(distortion.min()),
        "fisheye_distortion_max": float(distortion.max()),
    }
