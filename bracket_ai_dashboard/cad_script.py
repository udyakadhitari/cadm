r"""
generate_dataset.py  (v3: filleted gusset toes, zero-stress mid-side nodes filtered)
Gusset-supported L-bracket dataset generator using PyMAPDL (2D plane stress).

Usage:
    python generate_dataset.py converge   # 1) mesh convergence check on one design
    python generate_dataset.py generate   # 2) run all LHS samples (resumable)
    python generate_dataset.py merge      # 3) merge per-sample files into one .npz

Units: mm, N, MPa.
Geometry: wall at x = 0 (fixed), arm along +x, load on the tip face x = L.
Outline: (0,0) (L,0) (L,w) -> fillet(r) at gusset toe on arm -> diagonal
         -> fillet(r) at gusset toe on leg -> (w,H) (0,H)
Both gusset toes (w+g, w) and (w, w+g) are rounded with radius r.
"""
import sys
import time
from pathlib import Path

import numpy as np
from matplotlib.path import Path as MplPath
from scipy.interpolate import griddata
from scipy.stats import qmc

# ----------------------------- settings -----------------------------
OUT = Path("dataset2")      # new folder, keeps the old dataset untouched
N_SAMPLES = 1100          # a bit more than 1000 to absorb failed runs
SEED = 42
GRID = 128                # image resolution (GRID x GRID)
DOMAIN = (120.0, 120.0)   # fixed image window in mm (must cover max L and max H)

THICK = 10.0              # plate thickness, mm
E, NU = 200e3, 0.3        # steel, MPa
FORCE = 1000.0            # total tip force, N (kept constant)
ESIZE = 2.0               # element size, mm (confirm with 'converge')
WALL_STRIP = 5.0          # mm: nodes with x < WALL_STRIP are ignored for vm_max_far

NAMES = ["L", "H", "w", "g", "r", "theta_deg"]
LOW = [80.0, 80.0, 10.0, 5.0, 2.0, -45.0]
HIGH = [120.0, 120.0, 25.0, 40.0, 6.0, 45.0]
# validity: w+g <= 65 < min(L,H); fillet needs 2*t < g*sqrt(2) with t = r*tan(22.5deg):
# worst case r=6, g=5 -> 4.97 < 7.07 (ok).

SQ2 = np.sqrt(2.0)
TAN22 = np.tan(np.deg2rad(22.5))


# ----------------------------- sampling -----------------------------
def sample_params(n, seed=SEED):
    """Latin Hypercube sample of the design space, shape (n, 6)."""
    unit = qmc.LatinHypercube(d=len(NAMES), seed=seed).random(n)
    return qmc.scale(unit, LOW, HIGH)


# ----------------------------- geometry -----------------------------
def fillet_geometry(w, g, r):
    """Tangent points and centres of the two gusset-toe fillets (45 deg deflection)."""
    t = r * TAN22
    if 2 * t >= g * SQ2:
        raise ValueError(f"fillet too large for gusset: r={r:.2f}, g={g:.2f}")
    px, py = w + g, w          # toe on the arm
    qx, qy = w, w + g          # toe on the leg
    T1 = (px + t, w)
    T2 = (px - t / SQ2, w + t / SQ2)
    C1 = (px + t, w + r)
    T3 = (w + t / SQ2, qy - t / SQ2)
    T4 = (w, qy + t)
    C2 = (w + r, qy + t)
    return T1, T2, C1, T3, T4, C2


def arc_points(c, a, b, n=16):
    """Points along the short arc from a to b around centre c."""
    a0 = np.arctan2(a[1] - c[1], a[0] - c[0])
    b0 = np.arctan2(b[1] - c[1], b[0] - c[0])
    d = (b0 - a0 + np.pi) % (2 * np.pi) - np.pi
    ang = a0 + d * np.linspace(0, 1, n)
    rad = np.hypot(a[0] - c[0], a[1] - c[1])
    return [(c[0] + rad * np.cos(x), c[1] + rad * np.sin(x)) for x in ang]


def outline(L, H, w, g, r):
    """Polygon (list of vertices) used for the image mask."""
    T1, T2, C1, T3, T4, C2 = fillet_geometry(w, g, r)
    pts = [(0, 0), (L, 0), (L, w)]
    pts += arc_points(C1, T1, T2)
    pts += arc_points(C2, T3, T4)
    pts += [(w, H), (0, H)]
    return pts


# ----------------------------- MAPDL -----------------------------
def launch():
    from ansys.mapdl.core import launch_mapdl  # lazy import
    return launch_mapdl(nproc=2, override=True, loglevel="ERROR")


def solve_case(mapdl, L, H, w, g, r, theta_deg, esize=ESIZE):
    """Build, mesh, solve one design. Returns node coords, von Mises, |u|."""
    L, H, w, g, r, theta_deg = (float(v) for v in (L, H, w, g, r, theta_deg))
    T1, T2, C1, T3, T4, C2 = fillet_geometry(w, g, r)

    mapdl.clear()
    mapdl.prep7()

    # element + material
    mapdl.et(1, "PLANE183")
    mapdl.keyopt(1, 3, 3)             # plane stress with thickness
    mapdl.r(1, THICK)
    mapdl.mp("EX", 1, E)
    mapdl.mp("PRXY", 1, NU)

    # keypoints 1..9 on the outline, 10 and 11 = fillet centres
    kps = [(0, 0), (L, 0), (L, w), T1, T2, T3, T4, (w, H), (0, H)]
    for i, (x, y) in enumerate(kps, start=1):
        mapdl.k(i, float(x), float(y), 0)
    mapdl.k(10, float(C1[0]), float(C1[1]), 0)
    mapdl.k(11, float(C2[0]), float(C2[1]), 0)

    # lines 1..9 in order, then one area
    mapdl.l(1, 2)            # 1  bottom of arm
    mapdl.l(2, 3)            # 2  arm tip face (load)
    mapdl.l(3, 4)            # 3  arm top surface
    mapdl.larc(4, 5, 10, r)  # 4  fillet at toe on arm
    mapdl.l(5, 6)            # 5  gusset diagonal
    mapdl.larc(6, 7, 11, r)  # 6  fillet at toe on leg
    mapdl.l(7, 8)            # 7  inner face of leg
    mapdl.l(8, 9)            # 8  top of leg
    mapdl.l(9, 1)            # 9  wall face
    mapdl.al(*range(1, 10))

    # mesh (same element size for every sample)
    mapdl.esize(esize)
    mapdl.amesh("ALL")

    # boundary conditions: wall face x = 0 fixed
    mapdl.nsel("S", "LOC", "X", 0)
    mapdl.d("ALL", "ALL", 0)

    # load: total force FORCE spread equally over nodes on the tip face x = L
    mapdl.nsel("S", "LOC", "X", L)
    n = mapdl.mesh.n_node
    th = np.deg2rad(theta_deg)
    mapdl.f("ALL", "FX", FORCE * np.sin(th) / n)
    mapdl.f("ALL", "FY", -FORCE * np.cos(th) / n)

    # solve
    mapdl.allsel()
    mapdl.run("/SOLU")
    mapdl.antype("STATIC")
    mapdl.solve()
    mapdl.finish()

    # post-process
    mapdl.post1()
    mapdl.set(1, 1)
    mapdl.allsel()
    vm = np.asarray(mapdl.post_processing.nodal_eqv_stress())
    u = np.asarray(mapdl.post_processing.nodal_displacement("NORM"))
    xyz = np.asarray(mapdl.mesh.nodes)
    mapdl.finish()

    if not (len(vm) == len(u) == len(xyz)):
        raise RuntimeError(f"size mismatch: vm={len(vm)} u={len(u)} xyz={len(xyz)}")
    if not np.isfinite(vm).all():
        raise RuntimeError("non-finite stress values")
    # PLANE183 mid-side nodes come back with exactly 0 stress (read-out artefact).
    # Keep only nodes that carry a real value.
    keep = vm > 0
    if keep.sum() < 0.25 * len(vm):
        raise RuntimeError(f"too few valid nodes: {int(keep.sum())} of {len(vm)}")
    return xyz[keep], vm[keep], u[keep]


# ----------------------------- rasterization -----------------------------
def rasterize(xyz, vm, pts):
    """Interpolate nodal stress onto a fixed GRID x GRID image.
    Row index = y (row 0 is y = 0), column index = x. Plot with origin='lower'."""
    xs = np.linspace(0, DOMAIN[0], GRID)
    ys = np.linspace(0, DOMAIN[1], GRID)
    X, Y = np.meshgrid(xs, ys)
    inside = MplPath(pts).contains_points(np.c_[X.ravel(), Y.ravel()]).reshape(GRID, GRID)
    lin = griddata(xyz[:, :2], vm, (X, Y), method="linear")
    near = griddata(xyz[:, :2], vm, (X, Y), method="nearest")
    field = np.where(np.isnan(lin), near, lin)   # nearest-node fallback at the edges
    field = np.where(inside, field, 0.0)
    return inside.astype(np.uint8), field.astype(np.float32)


def vm_far(xyz, vm):
    """Max von Mises ignoring a thin strip next to the fixed wall (BC singularity)."""
    far = xyz[:, 0] > WALL_STRIP
    return vm[far].max()


# ----------------------------- drivers -----------------------------
def converge():
    mapdl = launch()
    mid = [(a + b) / 2 for a, b in zip(LOW, HIGH)]
    print("design (L,H,w,g,r,theta):", mid)
    print("esize  nodes  max_vm  max_vm(x>5)  max_u(mm)  time(s)  location of max")
    for es in [4.0, 3.0, 2.0, 1.5, 1.0, 0.75]:
        t0 = time.perf_counter()
        xyz, vm, u = solve_case(mapdl, *mid, esize=es)
        k = vm.argmax()
        print(f"{es:5.2f} {len(vm):6d} {vm.max():8.2f} {vm_far(xyz, vm):11.2f} "
              f"{u.max():10.4f} {time.perf_counter() - t0:8.2f}  x={xyz[k, 0]:.1f} y={xyz[k, 1]:.1f}")
    mapdl.exit()


def generate():
    OUT.mkdir(exist_ok=True)
    params = sample_params(N_SAMPLES)
    np.save(OUT / "params_all.npy", params)
    mapdl = launch()
    for i, p in enumerate(params):
        f = OUT / f"sample_{i:04d}.npz"
        if f.exists():                      # resume support
            continue
        L, H, w, g, r, th = p
        try:
            t0 = time.perf_counter()
            xyz, vm, u = solve_case(mapdl, L, H, w, g, r, th)
            t_fea = time.perf_counter() - t0   # build+mesh+solve+post
            mask, field = rasterize(xyz, vm, outline(L, H, w, g, r))
            np.savez_compressed(
                f, params=p, mask=mask, vm=field,
                vm_max=vm.max(), vm_max_far=vm_far(xyz, vm),
                u_max=u.max(), t_fea=t_fea, n_nodes=len(vm),
                nodes_xy=xyz[:, :2].astype(np.float32),   # raw data, kept for later use
                nodes_vm=vm.astype(np.float32),
            )
            if i % 25 == 0:
                print(f"[{i}/{len(params)}] vm_max={vm.max():.1f} MPa  t={t_fea:.2f}s")
        except Exception as e:
            print(f"[{i}] FAILED: {e}")
            with open(OUT / "failed.txt", "a") as fh:
                fh.write(f"{i}\t{e}\n")
            try:
                mapdl.exit()
            except Exception:
                pass
            mapdl = launch()
    mapdl.exit()
    print("done. now run: python generate_dataset.py merge")


def merge():
    files = sorted(OUT.glob("sample_*.npz"))
    data = [np.load(f) for f in files]
    keys = ["params", "mask", "vm", "vm_max", "vm_max_far", "u_max", "t_fea", "n_nodes"]
    merged = {k: np.stack([d[k] for d in data]) for k in keys}
    np.savez_compressed(OUT / "bracket_dataset.npz", **merged)
    print(f"merged {len(files)} samples -> {OUT / 'bracket_dataset.npz'}")
    print("mean FEA time per sample: %.2f s" % merged["t_fea"].mean())


if __name__ == "__main__":
    modes = {"converge": converge, "generate": generate, "merge": merge}
    if len(sys.argv) != 2 or sys.argv[1] not in modes:
        print("usage: python generate_dataset.py [converge|generate|merge]")
        sys.exit(1)
    modes[sys.argv[1]]()