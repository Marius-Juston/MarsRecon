import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, box, Point, MultiPoint, LineString
from shapely.affinity import translate, rotate
from scipy.optimize import brentq


# --- 1. CORE PIPELINE (From previous step) ---

def get_optimized_angles(target_polygon: Polygon, centroid: Point, extra_rays_per_edge: int = 2) -> np.ndarray:
    cx, cy = centroid.x, centroid.y
    coords = list(target_polygon.exterior.coords)[:-1]
    angles = []
    for i in range(len(coords)):
        vx, vy = coords[i]
        angle_to_vertex = np.arctan2(vy - cy, vx - cx) % (2 * np.pi)
        angles.append(angle_to_vertex)

        next_vx, next_vy = coords[(i + 1) % len(coords)]
        angle_to_next = np.arctan2(next_vy - cy, next_vx - cx) % (2 * np.pi)
        if angle_to_next < angle_to_vertex:
            angle_to_next += 2 * np.pi

        step = (angle_to_next - angle_to_vertex) / (extra_rays_per_edge + 1)
        for j in range(1, extra_rays_per_edge + 1):
            angles.append((angle_to_vertex + j * step) % (2 * np.pi))
    return np.sort(np.unique(angles))


def generate_valid_center_region(target_polygon: Polygon, square_size: float, overlap_percentage: float,
                                 extra_rays_per_edge: int = 2) -> Polygon:
    target_area = (square_size ** 2) * overlap_percentage
    half_L = square_size / 2.0
    base_square = box(-half_L, -half_L, half_L, half_L)

    def area_difference(r: float, theta: float, cx: float, cy: float) -> float:
        test_x = cx + r * np.cos(theta)
        test_y = cy + r * np.sin(theta)
        translated_square = translate(base_square, xoff=test_x, yoff=test_y)
        return target_polygon.intersection(translated_square).area - target_area

    centroid = target_polygon.centroid
    x0, y0 = centroid.x, centroid.y

    if area_difference(0, 0, x0, y0) < 0:
        return Polygon()  # Return empty polygon if impossible

    coords = np.array(target_polygon.exterior.coords)
    r_max = np.max(np.linalg.norm(coords - [x0, y0], axis=1)) + (square_size * np.sqrt(2))

    angles = get_optimized_angles(target_polygon, centroid, extra_rays_per_edge)
    boundary_points = []

    for theta in angles:
        try:
            r_star = brentq(area_difference, a=0, b=r_max, args=(theta, x0, y0))
            boundary_points.append((x0 + r_star * np.cos(theta), y0 + r_star * np.sin(theta)))
        except ValueError:
            pass

    return Polygon(boundary_points)


# --- 2. NEW BIN-PACKING ALGORITHM ---

def pack_squares_in_region(valid_region: Polygon, square_size: float, phase_steps: int = 10) -> list:
    """
    Finds the optimal grid phase to pack the maximum number of squares.

    Args:
        valid_region: The Shapely polygon representing valid center placements.
        square_size: The width/height of the squares being packed.
        phase_steps: Resolution of the grid-shift search. Higher is more optimal but slower.

    Returns:
        List of (x, y) tuples representing the optimal center points for the squares.
    """
    if valid_region.is_empty:
        return []

    minx, miny, maxx, maxy = valid_region.bounds
    best_points = []
    max_count = -1

    # We shift the grid between 0 and the square size to find the optimal "fit"
    dx_steps = np.linspace(0, square_size, phase_steps, endpoint=False)
    dy_steps = np.linspace(0, square_size, phase_steps, endpoint=False)

    for dx in dx_steps:
        for dy in dy_steps:
            # Generate the grid coordinates for this specific shift phase
            # We pad the max bounds slightly to ensure we don't miss edge points
            x_coords = np.arange(minx + dx, maxx + square_size, square_size)
            y_coords = np.arange(miny + dy, maxy + square_size, square_size)

            # Create a dense meshgrid
            xv, yv = np.meshgrid(x_coords, y_coords)

            # Convert to a Shapely MultiPoint for lightning-fast batch intersection
            grid_points = MultiPoint(np.column_stack([xv.ravel(), yv.ravel()]))

            # Filter down to ONLY the points inside the valid region
            valid_points = grid_points.intersection(valid_region)

            # Extract coordinates based on the geometry type Shapely returns
            if valid_points.is_empty:
                current_points = []
            elif valid_points.geom_type == 'Point':
                current_points = [(valid_points.x, valid_points.y)]
            elif valid_points.geom_type == 'MultiPoint':
                current_points = [(p.x, p.y) for p in valid_points.geoms]
            else:
                current_points = []

            # Update the best configuration if this phase shift packed more squares
            if len(current_points) > max_count:
                max_count = len(current_points)
                best_points = current_points

    return best_points


# --- 3. VISUALIZATION ---

def plot_polygon(ax, poly, **kwargs):
    """Helper to plot Shapely polygons on a matplotlib axis."""
    x, y = poly.exterior.xy
    ax.plot(x, y, **kwargs)
    ax.fill(x, y, alpha=0.2, color=kwargs.get('color', 'blue'))


def run_test_suite():
    # Define test parameters
    L = 4.0
    min_overlap = 0.50  # 50% overlap

    # Define a suite of convex shapes
    test_shapes = {
        "Standard Rectangle": rotate(Polygon([(0, 0), (10, 0), (10, 6), (0, 6)]), 45, origin='center'),
        "Chamfered Rect (Your Use Case)": Polygon([(1, 0), (9, 0), (10, 1), (10, 5), (9, 6), (1, 6), (0, 5), (0, 1)]),
        "Trapezoid": Polygon([(2, 0), (8, 0), (6, 6), (4, 6)]),
        "Hexagon": Polygon([(3, 0), (7, 0), (9, 4), (7, 8), (3, 8), (1, 4)])
    }

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f"Valid Center Regions (Square Size: {L}x{L}, Target Overlap: {min_overlap * 100}%)", fontsize=16)

    for ax, (title, base_poly) in zip(axes.flatten(), test_shapes.items()):
        ax.set_title(title)
        ax.set_aspect('equal')

        # 1. Plot Base Polygon
        plot_polygon(ax, base_poly, color='blue', label='Base Polygon', linewidth=2)

        try:
            # 2. Generate and Plot Valid Region
            # Using 3 extra rays per edge. For an 8-point chamfered rect, this is 8 * (1+3) = 32 rays.
            valid_region = generate_valid_center_region(base_poly, L, min_overlap, extra_rays_per_edge=3)
            plot_polygon(ax, valid_region, color='red', label='Valid Region for Center', linestyle='--', linewidth=2)

            # 3. Plot a Sample Square to prove the math
            # Grab a point on the boundary of the valid region
            sample_center = list(valid_region.exterior.coords)[0]
            half_L = L / 2.0
            sample_square = box(sample_center[0] - half_L, sample_center[1] - half_L,
                                sample_center[0] + half_L, sample_center[1] + half_L)

            # Plot the square outline and the intersection
            x, y = sample_square.exterior.xy
            ax.plot(x, y, color='green', linestyle=':', linewidth=2, label='Sample Square on Boundary')

            intersection = base_poly.intersection(sample_square)
            plot_polygon(ax, intersection, color='green')

            # Mark the exact center point
            ax.plot(sample_center[0], sample_center[1], 'ro', markersize=5)

        except Exception as e:
            ax.text(0.5, 0.5, f"Failed: {str(e)}", transform=ax.transAxes, ha='center', color='red')

        ax.legend(loc='upper right', fontsize='small')
        ax.grid(True, linestyle=':', alpha=0.6)

    plt.tight_layout()
    plt.savefig("min_overlap_polygon.png")
    plt.show()

def plot_packing(base_polygon: Polygon, valid_region: Polygon, packed_centers: list, square_size: float, id:int=0):
    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot Base Polygon
    x, y = base_polygon.exterior.xy
    ax.plot(x, y, color='blue', linewidth=2, label='Base Polygon')
    ax.fill(x, y, alpha=0.1, color='blue')

    # Plot Valid Center Region
    if not valid_region.is_empty:
        x, y = valid_region.exterior.xy
        ax.plot(x, y, color='red', linestyle='--', linewidth=2, label='Valid Center Region')

    # Plot the packed squares
    half_L = square_size / 2.0
    for i, (cx, cy) in enumerate(packed_centers):
        sq = box(cx - half_L, cy - half_L, cx + half_L, cy + half_L)
        x, y = sq.exterior.xy

        # Only add the label for the first square to avoid legend clutter
        label = 'Packed Squares' if i == 0 else None
        ax.plot(x, y, color='green', linewidth=1)
        ax.fill(x, y, alpha=0.3, color='green', label=label)
        ax.plot(cx, cy, 'k.', markersize=3)  # Mark center

    ax.set_aspect('equal')
    ax.set_title(f"Optimal Packing: {len(packed_centers)} Squares Packed", fontsize=14)
    ax.legend(loc='upper right')
    plt.grid(True, linestyle=':', alpha=0.5)
    plt.tight_layout()
    plt.savefig(f"packing_{id}.png")
    plt.show()

#TODO incorporate this into the hirise_sampler
def pack_squares_independent_strips(valid_region: Polygon, square_size: float, square_overlap: float = 0.0,
                                    phase_steps: int = 20) -> list:
    """
    Packs squares using decoupled horizontal and vertical strips, allowing squares to overlap each other.

    Args:
        valid_region: The Shapely polygon representing valid center placements.
        square_size: The side length of the squares (L).
        square_overlap: The percentage squares should overlap each other (0.0 to 0.99).
        phase_steps: Resolution of the alignment search.
    """
    if valid_region.is_empty:
        return []

    # Calculate the distance between centers based on required overlap
    stride = square_size * (1.0 - square_overlap)

    if stride <= 0:
        raise ValueError("Square overlap must be less than 1.0 (100%) to prevent infinite packing.")

    minx, miny, maxx, maxy = valid_region.bounds
    best_points = []
    max_count = 0

    # STRATEGY 1: Horizontal Rows
    # We sweep the Y-axis phase up to the new 'stride' instead of 'square_size'
    for dy in np.linspace(0, stride, phase_steps, endpoint=False):
        current_points = []
        y = miny + dy
        while y <= maxy:
            sweep_line = LineString([(minx - 1, y), (maxx + 1, y)])
            intersection = valid_region.intersection(sweep_line)

            if not intersection.is_empty and intersection.geom_type == 'LineString':
                ix_min, _, ix_max, _ = intersection.bounds
                available_width = ix_max - ix_min

                if available_width >= 0:
                    # Calculate count using the new STRIDE
                    count = int(available_width // stride) + 1
                    used_width = (count - 1) * stride
                    x_start = ix_min + (available_width - used_width) / 2.0

                    for i in range(count):
                        # Space points by STRIDE
                        current_points.append((x_start + i * stride, y))

            # Move up by STRIDE
            y += stride

        if len(current_points) > max_count:
            max_count = len(current_points)
            best_points = current_points

    # STRATEGY 2: Vertical Columns
    for dx in np.linspace(0, stride, phase_steps, endpoint=False):
        current_points = []
        x = minx + dx
        while x <= maxx:
            sweep_line = LineString([(x, miny - 1), (x, maxy + 1)])
            intersection = valid_region.intersection(sweep_line)

            if not intersection.is_empty and intersection.geom_type == 'LineString':
                _, iy_min, _, iy_max = intersection.bounds
                available_height = iy_max - iy_min

                if available_height >= 0:
                    count = int(available_height // stride) + 1
                    used_height = (count - 1) * stride
                    y_start = iy_min + (available_height - used_height) / 2.0

                    for i in range(count):
                        current_points.append((x, y_start + i * stride))
            x += stride

        if len(current_points) > max_count:
            max_count = len(current_points)
            best_points = current_points

    return best_points


if __name__ == "__main__":
    run_test_suite()
    # Define a LARGE base polygon (chamfered rectangle approximation)
    p_coords = [(2, 0), (28, 0), (30, 2), (30, 18), (28, 20), (2, 20), (0, 18), (0, 2)]
    large_polygon = rotate(Polygon(p_coords), 25, origin="center")

    # Parameters
    L = 1.5  # Small squares
    min_overlap = 0.60  # Require 60% overlap

    print("1. Calculating Valid Center Region...")
    valid_region = generate_valid_center_region(large_polygon, L, min_overlap, extra_rays_per_edge=3)

    print("2. Optimizing Grid Packing (Sweeping Phases)...")
    # A phase_steps of 10 means checking 10x10 = 100 different grid alignments
    optimal_centers = pack_squares_in_region(valid_region, L, phase_steps=10)

    print(f"Done. Packed {len(optimal_centers)} squares.")

    # Plot the result
    plot_packing(large_polygon, valid_region, optimal_centers, L, id=0)

    print("3. Optimizing Independent Strips...")
    # Because we dropped the grid, this is O(N) instead of O(N^2),
    # so we can easily run 30 phase steps for a hyper-optimized fit.
    optimal_centers = pack_squares_independent_strips(valid_region, L, phase_steps=30)

    print(f"Done. Packed {len(optimal_centers)} squares.")
    plot_packing(large_polygon, valid_region, optimal_centers, L, id=1)

    print("3. Optimizing Independent Strips... 50% stride")
    # Because we dropped the grid, this is O(N) instead of O(N^2),
    # so we can easily run 30 phase steps for a hyper-optimized fit.
    optimal_centers = pack_squares_independent_strips(valid_region, L, phase_steps=30, square_overlap=0.1)

    print(f"Done. Packed {len(optimal_centers)} squares.")
    plot_packing(large_polygon, valid_region, optimal_centers, L, id=2)