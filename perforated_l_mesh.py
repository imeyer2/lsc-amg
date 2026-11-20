"""
Perforated L-domain mesh generator using meshpy (Triangle).

Produces a constrained Delaunay triangulation of an L-shaped domain
with circular holes.  The mesh intentionally includes:
  - A re-entrant corner at (0.5, 0.5)  →  solution singularity ~ r^{2/3}
  - Multiple circular holes             →  topological obstacles for AMG
  - At least one hole near the corner    →  worst-case interaction
  - At least one narrow bridge           →  high aspect-ratio elements
  - Controllable mesh quality            →  lower min-angle = worse elements

Returns data compatible with scikit-fem MeshTri.
"""

import numpy as np
import meshpy.triangle as triangle


# ── L-shaped domain boundary ────────────────────────────────────
#  (0,1)───────(0.5,1)
#    │             │
#    │             │
#    │      (0.5,0.5)──────(1,0.5)
#    │                        │
#    │                        │
#  (0,0)──────────────────(1,0)

L_VERTICES = np.array([
    [0.0, 0.0],
    [1.0, 0.0],
    [1.0, 0.5],
    [0.5, 0.5],
    [0.5, 1.0],
    [0.0, 1.0],
])


def _point_in_l_domain(x, y, margin=0.0):
    """Check if (x,y) lies inside the L-shaped domain with optional margin."""
    in_bottom = (x > margin) & (x < 1.0 - margin) & (y > margin) & (y < 0.5 - margin)
    in_left   = (x > margin) & (x < 0.5 - margin) & (y > margin) & (y < 1.0 - margin)
    return in_bottom | in_left


def _circles_overlap(centers, radii, new_c, new_r, min_gap):
    """Check if a new circle overlaps any existing circle (with min gap)."""
    if len(centers) == 0:
        return False
    dists = np.linalg.norm(centers - new_c, axis=1)
    return np.any(dists < radii + new_r + min_gap)


def _circle_in_l_domain(cx, cy, r, margin=0.0):
    """Check if a circle fits entirely inside the L domain."""
    # The circle must have every point inside the L.
    # For the L-domain, check that the circle's bounding box corners are inside,
    # PLUS that the circle doesn't cross the re-entrant corner notch.
    m = margin + r
    # Must fit in bottom rectangle OR left rectangle
    in_bottom = (cx - r > margin) and (cx + r < 1.0 - margin) and \
                (cy - r > margin) and (cy + r < 0.5 - margin)
    in_left   = (cx - r > margin) and (cx + r < 0.5 - margin) and \
                (cy - r > margin) and (cy + r < 1.0 - margin)
    return in_bottom or in_left


def place_holes(rng, n_holes=5, radius_range=(0.02, 0.06), margin=0.01,
                force_corner_hole=True, force_close_pair=True):
    """
    Place n_holes non-overlapping circles inside the L domain.

    Parameters
    ----------
    rng : np.random.Generator
    n_holes : int
    radius_range : (float, float), min and max hole radius
    margin : float, minimum distance from boundary
    force_corner_hole : bool, place one hole near (0.5, 0.5) re-entrant corner
    force_close_pair : bool, place two holes close together (thin bridge)

    Returns
    -------
    centers : (n_holes, 2)
    radii   : (n_holes,)
    """
    centers = []
    radii_list = []
    rmin, rmax = radius_range
    min_gap = 0.008  # minimum gap between holes (thin bridge when ~= this)

    placed = 0

    # --- Force a hole near the re-entrant corner (0.5, 0.5) ---
    if force_corner_hole and n_holes >= 1:
        for _ in range(500):
            r = rng.uniform(rmin, rmax * 0.7)  # slightly smaller near corner
            # Place in the bottom-rect side near (0.5, 0.5)
            cx = rng.uniform(0.5 - 0.12, 0.5 - r - margin)
            cy = rng.uniform(0.5 - 0.12, 0.5 - r - margin)
            if _circle_in_l_domain(cx, cy, r, margin):
                centers.append([cx, cy])
                radii_list.append(r)
                placed += 1
                break

    # --- Force a close pair (thin bridge) ---
    if force_close_pair and n_holes >= placed + 2:
        for _ in range(500):
            r1 = rng.uniform(rmin, rmax)
            r2 = rng.uniform(rmin, rmax)
            # Place first circle somewhere in the bottom rectangle
            cx1 = rng.uniform(0.15, 0.85)
            cy1 = rng.uniform(0.08, 0.35)
            if not _circle_in_l_domain(cx1, cy1, r1, margin):
                continue
            if _circles_overlap(np.array(centers) if centers else np.empty((0, 2)),
                                np.array(radii_list) if radii_list else np.empty(0),
                                np.array([cx1, cy1]), r1, min_gap):
                continue
            # Place second circle very close to first
            gap = rng.uniform(min_gap, min_gap * 3)
            angle = rng.uniform(0, 2 * np.pi)
            cx2 = cx1 + (r1 + r2 + gap) * np.cos(angle)
            cy2 = cy1 + (r1 + r2 + gap) * np.sin(angle)
            if not _circle_in_l_domain(cx2, cy2, r2, margin):
                continue
            all_c = np.array(centers + [[cx1, cy1]]) if centers else np.array([[cx1, cy1]])
            all_r = np.array(radii_list + [r1]) if radii_list else np.array([r1])
            if _circles_overlap(all_c, all_r, np.array([cx2, cy2]), r2, min_gap * 0.5):
                continue
            centers.extend([[cx1, cy1], [cx2, cy2]])
            radii_list.extend([r1, r2])
            placed += 2
            break

    # --- Fill remaining holes randomly ---
    attempts = 0
    while placed < n_holes and attempts < 5000:
        attempts += 1
        r = rng.uniform(rmin, rmax)
        cx = rng.uniform(r + margin, 1.0 - r - margin)
        cy = rng.uniform(r + margin, 1.0 - r - margin)
        if not _circle_in_l_domain(cx, cy, r, margin):
            continue
        if _circles_overlap(np.array(centers) if centers else np.empty((0, 2)),
                            np.array(radii_list) if radii_list else np.empty(0),
                            np.array([cx, cy]), r, min_gap):
            continue
        centers.append([cx, cy])
        radii_list.append(r)
        placed += 1

    centers = np.array(centers)
    radii_arr = np.array(radii_list)
    return centers, radii_arr


def build_perforated_l_mesh(rng, n_holes=5, radius_range=(0.02, 0.06),
                            max_area=None, min_angle=20.0,
                            n_circle_segments=24,
                            force_corner_hole=True,
                            force_close_pair=True):
    """
    Build a triangulation of the perforated L-domain.

    Parameters
    ----------
    rng : np.random.Generator
    n_holes : int
    radius_range : (float, float)
    max_area : float or None
        Maximum triangle area.  None → auto from n_holes/domain size.
    min_angle : float
        Minimum angle constraint (degrees).  Lower = worse mesh quality.
        20 = noticeably poor,  30 = good quality,  15 = very poor.
    n_circle_segments : int
        Number of line segments per hole boundary circle.
    force_corner_hole : bool
    force_close_pair : bool

    Returns
    -------
    points : (N, 2)        node coordinates
    triangles : (T, 3)     triangle connectivity (0-based)
    boundary_nodes : set    indices of nodes on outer boundary or hole boundaries
    hole_centers : (H, 2)  for reference
    hole_radii : (H,)      for reference
    """
    centers, radii = place_holes(
        rng, n_holes=n_holes, radius_range=radius_range,
        force_corner_hole=force_corner_hole,
        force_close_pair=force_close_pair,
    )
    actual_n_holes = len(centers)

    if max_area is None:
        # Auto: aim for ~2000-4000 triangles
        max_area = 0.75 / 2500  # L-domain area ≈ 0.75

    # ── Build PSLG ──
    info = triangle.MeshInfo()

    # -- Outer boundary (L-shape) --
    n_outer = len(L_VERTICES)
    all_points = list(map(tuple, L_VERTICES))
    facets = [(i, (i + 1) % n_outer) for i in range(n_outer)]

    # -- Hole boundaries --
    hole_points_list = []  # list of (center, radius, start_idx)
    for hi in range(actual_n_holes):
        cx, cy = centers[hi]
        r = radii[hi]
        start_idx = len(all_points)
        theta = np.linspace(0, 2 * np.pi, n_circle_segments, endpoint=False)
        for t in theta:
            all_points.append((cx + r * np.cos(t), cy + r * np.sin(t)))
        # Close the loop
        for j in range(n_circle_segments):
            i0 = start_idx + j
            i1 = start_idx + (j + 1) % n_circle_segments
            facets.append((i0, i1))
        hole_points_list.append((start_idx, n_circle_segments))

    info.set_points(all_points)
    info.set_facets(facets)

    # Tell Triangle where holes are (one interior point per hole)
    if actual_n_holes > 0:
        info.set_holes([(float(c[0]), float(c[1])) for c in centers])

    # ── Triangulate ──
    mesh = triangle.build(
        info,
        max_volume=max_area,
        min_angle=min_angle,
        allow_boundary_steiner=True,
        allow_volume_steiner=True,
        generate_faces=True,
    )

    points = np.array(mesh.points)
    tris = np.array(mesh.elements)

    # ── Identify boundary nodes ──
    # Facets from Triangle output mark boundary edges
    facet_arr = np.array(mesh.facets)
    boundary_nodes = set(facet_arr.ravel())

    return points, tris, boundary_nodes, centers, radii


def mesh_quality_stats(points, triangles):
    """Compute mesh quality statistics."""
    stats = {}
    p = points[triangles]  # (T, 3, 2)
    # Edge vectors
    e0 = p[:, 1] - p[:, 0]
    e1 = p[:, 2] - p[:, 1]
    e2 = p[:, 0] - p[:, 2]

    # Edge lengths
    l0 = np.linalg.norm(e0, axis=1)
    l1 = np.linalg.norm(e1, axis=1)
    l2 = np.linalg.norm(e2, axis=1)

    # Angles via law of cosines
    def angle(a, b, c):
        """Angle opposite to side c."""
        cos_val = (a**2 + b**2 - c**2) / (2 * a * b + 1e-30)
        cos_val = np.clip(cos_val, -1, 1)
        return np.degrees(np.arccos(cos_val))

    a0 = angle(l0, l2, l1)  # angle at vertex 0
    a1 = angle(l0, l1, l2)  # angle at vertex 1
    a2 = angle(l1, l2, l0)  # angle at vertex 2
    all_angles = np.concatenate([a0, a1, a2])

    # Aspect ratio = longest edge / shortest altitude
    areas = 0.5 * np.abs(e0[:, 0] * e2[:, 1] - e0[:, 1] * e2[:, 0])
    longest_edge = np.maximum(l0, np.maximum(l1, l2))
    shortest_edge = np.minimum(l0, np.minimum(l1, l2))
    aspect_ratios = longest_edge / (2 * areas / (longest_edge + 1e-30) + 1e-30)

    stats['n_points'] = len(points)
    stats['n_triangles'] = len(triangles)
    stats['min_angle'] = float(all_angles.min())
    stats['max_angle'] = float(all_angles.max())
    stats['mean_angle'] = float(all_angles.mean())
    stats['min_area'] = float(areas.min())
    stats['max_area'] = float(areas.max())
    stats['area_ratio'] = float(areas.max() / (areas.min() + 1e-30))
    stats['min_aspect_ratio'] = float(aspect_ratios.min())
    stats['max_aspect_ratio'] = float(aspect_ratios.max())
    stats['mean_aspect_ratio'] = float(aspect_ratios.mean())
    stats['edge_length_ratio'] = float(longest_edge.max() / (shortest_edge.min() + 1e-30))

    return stats, areas, aspect_ratios, all_angles


if __name__ == "__main__":
    rng = np.random.default_rng(42)
    pts, tris, bd, centers, radii = build_perforated_l_mesh(
        rng, n_holes=6, min_angle=20.0
    )
    stats, _, _, _ = mesh_quality_stats(pts, tris)
    print(f"Mesh: {stats['n_points']} nodes, {stats['n_triangles']} triangles")
    print(f"Min angle:  {stats['min_angle']:.1f}°")
    print(f"Max aspect: {stats['max_aspect_ratio']:.1f}")
    print(f"Area ratio: {stats['area_ratio']:.1f}")
    print(f"Holes placed: {len(centers)}")
