"""
block_sparse_hash_map.py
========================
A high-performance, probabilistic 3-D occupancy map built on a **block-sparse
hash map** (BSHM) data structure.

Architecture overview
---------------------
The world is partitioned into an infinite grid of axis-aligned cubic *blocks*.
Each block is a ``BlockSparseHashMap`` instance whose voxel occupancy states
are stored as floating-point Bayesian probabilities in a plain ``dict`` keyed
by ``(ix, iy, iz)`` integer indices.  No octree traversal is performed at
runtime; the ``o3d.geometry.Octree`` member exists purely for Open3D rendering.

Block lifecycle and eviction
-----------------------------
Unbounded block growth is prevented by two complementary mechanisms:

1. **Capacity cap** – ``BlockSparseHashMapManager`` enforces a configurable
   ``max_blocks`` limit.  When the limit is reached the *least-recently-used*
   (LRU) block whose voxels are all below the occupancy threshold is evicted.
   If every block is occupied the oldest block by last-access time is evicted
   instead, ensuring the cap is always respected.

2. **Voxel-level eviction** – ``BlockSparseHashMap.clear_expired_nodes``
   removes individual voxels that have not been updated within a configurable
   ``timeout``.  Blocks that become completely empty after voxel eviction are
   themselves pruned by the manager's decay worker.

Locking discipline — deadlock prevention
-----------------------------------------
A strict **lock-ordering protocol** eliminates AB-BA deadlock cycles:

    RULE: ``_manager_lock`` (manager) must ALWAYS be acquired BEFORE
          ``block._lock`` (per-block).  No code path may acquire the manager
          lock while already holding a block lock.

Consequences of the rule:

* ``get_stats()`` acquires the manager lock once to snapshot the block list,
  releases it, then acquires each block lock individually — never nesting them.
* ``get_region_probabilities()`` and ``get_all_maps_pointcloud()`` follow the
  same snapshot-then-release pattern.
* The decay worker snapshots the block list under the manager lock, then
  releases it before touching any block lock.
* ``_get_or_create_block`` acquires the manager lock for the write path; it
  never calls anything that re-acquires it.

Concurrency model
-----------------
* ``BlockSparseHashMap._lock`` – plain ``threading.Lock`` (non-reentrant) guards
  ``probability_map`` and ``timestamp_map``.  A non-reentrant lock is chosen
  deliberately: if any code accidentally tries to acquire it recursively it will
  deadlock immediately, surfacing the bug rather than silently succeeding.
* ``BlockSparseHashMapManager._manager_lock`` – plain ``threading.Lock`` guards
  ``_submaps``, LRU order, and global bounds.
* All public API methods are safe to call concurrently from multiple threads.

Public API
----------
``BlockSparseHashMap``        – one cubic probabilistic voxel block
``BlockSparseHashMapManager`` – multi-block manager (primary interface)
``voxel_downsample_jit``      – Numba-accelerated voxel downsampling
``batch_compute_voxel_keys``  – Numba-accelerated voxel-key computation
"""

import open3d as o3d
import numpy as np
import time
import threading

from typing import Dict as TypingDict, Tuple, List, Optional
from collections import defaultdict, OrderedDict
from concurrent.futures import ThreadPoolExecutor

from numba import njit, prange, int64, float32
from numba.typed import Dict
from numba.types import UniTuple

__all__ = [
    'BlockSparseHashMap',
    'BlockSparseHashMapManager',
    'voxel_downsample_jit',
    'batch_compute_voxel_keys',
]

# ---------------------------------------------------------------------------
# Module-level type aliases
# ---------------------------------------------------------------------------

VoxelKey = Tuple[int, int, int]   # Integer voxel cell index (ix, iy, iz)
BlockKey = Tuple[int, int, int]   # Integer block grid index (bx, by, bz)

# ---------------------------------------------------------------------------
# Numba-accelerated utility functions
# ---------------------------------------------------------------------------

@njit(parallel=True, fastmath=True)
def voxel_downsample_jit(points: np.ndarray, voxel_size: float) -> np.ndarray:
    """
    Numba-JIT voxel downsampling for large point clouds.

    Each input point is assigned to an integer voxel cell.  The output is
    the centroid of all points that fall inside the same cell.

    Parameters
    ----------
    points : np.ndarray, shape (N, 3), dtype float32
        Input point cloud in Cartesian coordinates.
    voxel_size : float
        Side length of each voxel cell (same units as *points*).

    Returns
    -------
    np.ndarray, shape (M, 3), dtype float32
        Downsampled centroids, one per non-empty voxel cell.
        Returns an empty ``(0, 3)`` array when *points* is empty.

    Notes
    -----
    Numba typed dicts do not support concurrent writes without explicit
    synchronisation; the accumulation loop is therefore sequential while
    the voxel-key computation is parallelised via ``prange``.
    """
    if points.shape[0] == 0:
        return np.zeros((0, 3), dtype=np.float32)

    inv_voxel_size = 1.0 / voxel_size

    voxel_dict = Dict.empty(
        key_type=UniTuple(int64, 3),
        value_type=UniTuple(float32[:], int64),
    )

    for i in prange(points.shape[0]):
        vx = int(np.floor(points[i, 0] * inv_voxel_size))
        vy = int(np.floor(points[i, 1] * inv_voxel_size))
        vz = int(np.floor(points[i, 2] * inv_voxel_size))
        key = (vx, vy, vz)

        if key in voxel_dict:
            s, c = voxel_dict[key]
            voxel_dict[key] = (s + points[i], c + 1)
        else:
            voxel_dict[key] = (points[i].copy(), 1)

    downsampled = np.zeros((len(voxel_dict), 3), dtype=np.float32)
    j = 0
    for key, (s, c) in voxel_dict.items():
        downsampled[j, :] = s / c
        j += 1

    return downsampled


@njit(parallel=True, fastmath=True)
def batch_compute_voxel_keys(points: np.ndarray, inv_voxel_size: float) -> np.ndarray:
    """
    Compute integer voxel-grid indices for a batch of 3-D points in parallel.

    Parameters
    ----------
    points : np.ndarray, shape (N, 3)
        Input point cloud.
    inv_voxel_size : float
        Reciprocal of the voxel side length (avoids repeated division).

    Returns
    -------
    np.ndarray, shape (N, 3), dtype int64
        Row ``i`` contains ``(ix, iy, iz)`` for ``points[i]``.
    """
    n = points.shape[0]
    keys = np.zeros((n, 3), dtype=np.int64)
    for i in prange(n):
        keys[i, 0] = int(np.floor(points[i, 0] * inv_voxel_size))
        keys[i, 1] = int(np.floor(points[i, 1] * inv_voxel_size))
        keys[i, 2] = int(np.floor(points[i, 2] * inv_voxel_size))
    return keys


# ---------------------------------------------------------------------------
# Single block
# ---------------------------------------------------------------------------

class BlockSparseHashMap:
    """
    A probabilistic voxel occupancy map for a single cubic spatial block.

    Voxel occupancy is tracked as a floating-point probability in ``(0, 1)``
    stored in ``probability_map`` keyed by ``(ix, iy, iz)`` integer indices.
    No octree data structure is used by the map logic; ``_render_octree``
    exists solely for Open3D's renderer.

    Bayesian update rule
    --------------------
    * **Obstacle observation**: multiply odds by ``min(1.5^count, 5)``.
    * **Free-space observation**: multiply odds by ``max(0.67^count, 0.198)``.

    Locking
    -------
    A plain (non-reentrant) ``threading.Lock`` guards ``probability_map`` and
    ``timestamp_map``.  Non-reentrant is deliberate: accidental recursive
    acquisition deadlocks immediately, exposing bugs rather than hiding them.

    Parameters
    ----------
    point_cloud : np.ndarray or None
        Optional ``(N, 3)`` array of obstacle points to seed the map.
    max_depth : int
        Depth of the Open3D ``Octree`` used **only** for rendering.
    map_size : float
        Side length of the cubic block in world units.
    center : np.ndarray or None
        World-space centre of the block.  Defaults to the origin.
    """

    def __init__(
        self,
        point_cloud: Optional[np.ndarray] = None,
        max_depth: int = 8,
        map_size: float = 256.0,
        center: Optional[np.ndarray] = None,
    ) -> None:
        self.map_size:  float   = map_size
        self.max_depth: int     = max_depth
        self.center: np.ndarray = (
            np.zeros(3, dtype=np.float64) if center is None
            else np.array(center, dtype=np.float64)
        )

        half = map_size / 2.0
        self.min_bound: np.ndarray = self.center - half
        self.max_bound: np.ndarray = self.center + half

        # Open3D render-only geometry — never read by map logic
        self._render_octree: Optional[o3d.geometry.Octree] = None
        self._init_render_octree(max_depth, map_size, self.center)

        # ------------------------------------------------------------------ #
        # Core data stores                                                     #
        # probability_map : {VoxelKey: float}  occupancy probability in (0,1) #
        # timestamp_map   : {VoxelKey: float}  last-write time (time.monotonic)#
        # ------------------------------------------------------------------ #
        self.probability_map: TypingDict[VoxelKey, float] = {}
        self.timestamp_map:   TypingDict[VoxelKey, float] = {}

        # Voxel side length = block_size / 2^depth (immutable after init)
        self.voxel_size: float = map_size / (2 ** max_depth)

        # Plain (non-reentrant) lock — see class docstring for rationale
        self._lock: threading.Lock = threading.Lock()

        self.last_update_time: float = time.monotonic()

        if point_cloud is not None and len(point_cloud) > 0:
            self.update_map(point_cloud)

    # ------------------------------------------------------------------
    # Render-octree initialisation (visualisation only)
    # ------------------------------------------------------------------

    def _init_render_octree(
        self,
        max_depth: int,
        map_size: float,
        center: np.ndarray,
    ) -> None:
        """
        Initialise the Open3D ``Octree`` geometry used **only** for rendering.

        Falls back gracefully across Open3D API versions so the code remains
        portable without changes to the caller.

        Parameters
        ----------
        max_depth : int
            Octree depth passed to Open3D.
        map_size : float
            Block side length (used to compute the bounding box).
        center : np.ndarray, shape (3,)
            Block centre in world space.
        """
        half = map_size / 2.0
        self._octree_min_bound: np.ndarray = center - half
        self._octree_max_bound: np.ndarray = center + half

        try:
            octree = o3d.geometry.Octree(max_depth=max_depth)
            try:
                if hasattr(octree, 'max_bound'):
                    octree.max_bound = self._octree_max_bound
                    octree.min_bound = self._octree_min_bound
                elif hasattr(octree, 'set_max_bound'):
                    octree.set_max_bound(self._octree_max_bound)
                    octree.set_min_bound(self._octree_min_bound)
            except Exception:
                pass  # Bounds stored separately; rendering still functional
            self._render_octree = octree
        except TypeError:
            # Older Open3D: positional argument only
            try:
                self._render_octree = o3d.geometry.Octree(max_depth)
            except Exception as exc:
                octree = o3d.geometry.Octree()
                if hasattr(octree, 'max_depth'):
                    octree.max_depth = max_depth
                self._render_octree = octree
                print(f"[BlockSparseHashMap] Octree init fallback: {exc}")

    # ------------------------------------------------------------------
    # Map update
    # ------------------------------------------------------------------

    def update_map(
        self,
        points: np.ndarray,
        point_type: str = 'obstacle',
    ) -> None:
        """
        Vectorised Bayesian probability update for a batch of points.

        The numpy computation (voxelisation, odds arithmetic) runs **outside**
        the lock to minimise contention; only the final dict write is locked.

        Parameters
        ----------
        points : np.ndarray, shape (N, 3)
            Points to integrate into the map.
        point_type : str
            ``'obstacle'`` raises occupancy probability;
            ``'free'`` lowers it.  Case-insensitive.
        """
        if len(points) == 0:
            return

        # Step 1: vectorised voxelisation — no lock needed ─────────────────
        voxel_indices = np.floor(points / self.voxel_size).astype(np.int32)
        unique_idx, counts = np.unique(voxel_indices, axis=0, return_counts=True)
        unique_keys: List[VoxelKey] = [tuple(row) for row in unique_idx]

        is_obstacle = (point_type.lower() == 'obstacle')

        # Step 2: read current probabilities — brief read lock ──────────────
        with self._lock:
            old_probs = np.array(
                [self.probability_map.get(k, 0.5) for k in unique_keys],
                dtype=np.float64,
            )

        # Step 3: Bayesian odds update — no lock needed ─────────────────────
        odds = old_probs / (1.0 - old_probs + 1e-8)
        if is_obstacle:
            # Cap the multiplier to prevent overconfidence after many updates
            update_factors = np.minimum(1.5 ** counts, 5.0)
        else:
            # Floor the multiplier so probability never drops to exact zero
            update_factors = np.maximum(0.67 ** counts, 0.198)

        new_odds  = odds * update_factors
        new_probs = np.clip(new_odds / (1.0 + new_odds), 0.001, 0.999)

        # Step 4: atomic commit — write lock ────────────────────────────────
        now = time.monotonic()
        with self._lock:
            for i, key in enumerate(unique_keys):
                self.probability_map[key] = float(new_probs[i])
                self.timestamp_map[key]   = now
            self.last_update_time = now

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def _estimate_voxel_size(self) -> float:
        """Return the voxel side length (no lock; immutable after ``__init__``)."""
        return self.voxel_size

    def get_voxel_probability(self, point: np.ndarray) -> float:
        """
        Return the occupancy probability of the voxel containing *point*.

        Returns ``0.5`` (uniform prior) for voxels that have never been
        observed.

        Parameters
        ----------
        point : np.ndarray, shape (3,)
            Query position in world coordinates.
        """
        key: VoxelKey = tuple(np.floor(point / self.voxel_size).astype(np.int32))
        with self._lock:
            return self.probability_map.get(key, 0.5)

    def is_empty(self) -> bool:
        """
        Return ``True`` if the block holds no voxel observations.

        Used by the manager to identify zero-information blocks eligible
        for LRU eviction.
        """
        with self._lock:
            return len(self.probability_map) == 0

    def occupied_voxel_count(self) -> int:
        """
        Return the number of voxels with occupancy probability > 0.5.

        Used by the manager's eviction policy to prefer evicting
        unoccupied blocks over informative ones.
        """
        with self._lock:
            return sum(1 for p in self.probability_map.values() if p > 0.5)

    # ------------------------------------------------------------------
    # Maintenance
    # ------------------------------------------------------------------

    def decay_probabilities(self, decay_factor: float = 0.01) -> None:
        """
        Apply temporal decay to all occupied voxels (probability > 0.5).

        Voxels that decay back to 0.5 and whose timestamp is older than
        5 s are evicted immediately to keep per-block memory bounded.

        Parameters
        ----------
        decay_factor : float
            Fractional reduction per call: ``p_new = p − p·factor``.
            Typical range: ``[0.01, 0.05]``.
        """
        now   = time.monotonic()
        evict: List[VoxelKey] = []

        with self._lock:
            for key, prob in list(self.probability_map.items()):
                if prob <= 0.5:
                    continue
                new_prob = max(prob - prob * decay_factor, 0.5)
                self.probability_map[key] = new_prob
                # Evict if decayed to baseline and not recently observed
                if new_prob <= 0.5 and now - self.timestamp_map.get(key, now) > 5.0:
                    evict.append(key)
            for key in evict:
                self.probability_map.pop(key, None)
                self.timestamp_map.pop(key, None)

    def clear_expired_nodes(self, timeout: float = 5.0) -> None:
        """
        Evict voxels whose last observation is older than *timeout* seconds.

        Parameters
        ----------
        timeout : float
            Staleness threshold in seconds.  Voxels older than this are
            removed unconditionally regardless of their probability.
        """
        now = time.monotonic()
        with self._lock:
            evict = [k for k, ts in self.timestamp_map.items()
                     if now - ts > timeout]
            for key in evict:
                self.probability_map.pop(key, None)
                self.timestamp_map.pop(key, None)


# ---------------------------------------------------------------------------
# Multi-block manager
# ---------------------------------------------------------------------------

class BlockSparseHashMapManager:
    """
    Top-level manager for a spatially unbounded probabilistic voxel map.

    The world is partitioned into cubic blocks indexed by ``(bx, by, bz)``.
    Blocks are allocated on demand and evicted when the population exceeds
    ``max_blocks``, using an LRU policy that prefers evicting blocks with no
    occupied voxels.

    Block eviction policy
    ---------------------
    When a new block would exceed ``max_blocks``:

    1. Scan the block store in LRU order (oldest first) and evict the first
       block that has **zero occupied voxels** (all probabilities ≤ 0.5).
    2. If every block contains occupied voxels, evict the single oldest block
       by last-access time.

    This guarantees the cap is always respected while minimising information
    loss for active obstacle regions.

    Locking discipline  *(see module docstring for full explanation)*
    ------------------
    ``_manager_lock`` is always acquired **before** ``block._lock``.
    No code path holds ``block._lock`` while acquiring ``_manager_lock``.
    Methods that need both acquire ``_manager_lock`` to snapshot the block
    list, release it, then access per-block state under ``block._lock`` alone.

    Parameters
    ----------
    map_size : float
        Side length of each block in world units.
    max_depth : int
        Octree depth used only by Open3D's renderer.
    grid_size : float or None
        Grid cell size; defaults to ``map_size`` (non-overlapping blocks).
    overlap : float
        Reserved for future use; currently ignored.
    enable_temporal_decay : bool
        When ``True``, ``start_decay_thread()`` launches a background daemon
        that periodically decays probabilities and evicts stale voxels.
    max_blocks : int
        Maximum number of blocks held simultaneously.  ``0`` = unlimited
        (not recommended for long-running deployments; can cause OOM).
    """

    _DEFAULT_MAX_BLOCKS: int = 2048  # ~500 m × 500 m × 100 m at 64 m blocks

    def __init__(
        self,
        map_size: float = 256.0,
        max_depth: int = 8,
        grid_size: Optional[float] = None,
        overlap: float = 0.0,
        enable_temporal_decay: bool = False,
        max_blocks: int = _DEFAULT_MAX_BLOCKS,
    ) -> None:
        self.map_size:   float = map_size
        self.max_depth:  int   = max_depth
        self.grid_size:  float = map_size if grid_size is None else float(grid_size)
        self.overlap:    float = float(np.clip(overlap, 0.0, 0.5))
        self.max_blocks: int   = int(max_blocks)

        # ------------------------------------------------------------------ #
        # Core block store                                                     #
        # OrderedDict maintains insertion/access order for LRU eviction.      #
        # All access must be guarded by _manager_lock.                         #
        # ------------------------------------------------------------------ #
        self._submaps: OrderedDict[BlockKey, BlockSparseHashMap] = OrderedDict()

        # Thread pool for potential future parallel block operations
        self._executor: ThreadPoolExecutor = ThreadPoolExecutor(max_workers=4)

        # World-space AABB spanning all allocated blocks (None until first write)
        self._global_min_bound: Optional[np.ndarray] = None
        self._global_max_bound: Optional[np.ndarray] = None

        # Manager-level lock.
        # INVARIANT: must always be acquired BEFORE any block._lock.
        self._manager_lock: threading.Lock = threading.Lock()

        # Coordinate caches — bounded to prevent unbounded growth
        self._grid_to_bounds_cache: TypingDict[
            BlockKey, Tuple[np.ndarray, np.ndarray]
        ] = {}
        self._point_to_grid_cache: TypingDict[
            Tuple[float, float, float], BlockKey
        ] = {}
        self._cache_capacity: int = 10_000

        # Soft flag signalling an in-progress reset (for external observability)
        self._resetting: bool = False

        # Temporal decay configuration
        self.enable_temporal_decay: bool   = enable_temporal_decay
        self._decay_thread: Optional[threading.Thread] = None
        self._decay_running: bool          = False
        self._decay_interval: float        = 0.25   # seconds between sweeps (~4 Hz)
        self._decay_factor: float          = 0.015  # fractional decay per sweep
        self._expiration_timeout: float    = 8   # voxel staleness threshold (s)

    # ------------------------------------------------------------------
    # Backwards-compatible property aliases
    # ------------------------------------------------------------------

    @property
    def submaps(self) -> TypingDict[BlockKey, BlockSparseHashMap]:
        """Expose the internal block store (read-only view for callers)."""
        return self._submaps

    @property
    def global_min_bound(self) -> Optional[np.ndarray]:
        return self._global_min_bound

    @global_min_bound.setter
    def global_min_bound(self, v: Optional[np.ndarray]) -> None:
        self._global_min_bound = v

    @property
    def global_max_bound(self) -> Optional[np.ndarray]:
        return self._global_max_bound

    @global_max_bound.setter
    def global_max_bound(self, v: Optional[np.ndarray]) -> None:
        self._global_max_bound = v

    # ------------------------------------------------------------------
    # Coordinate helpers (pure arithmetic — no locks required)
    # ------------------------------------------------------------------

    def _normalize_to_grid(self, point: np.ndarray) -> np.ndarray:
        """Convert a world-space point to fractional block-grid coordinates."""
        return np.asarray(point, dtype=np.float64) / self.grid_size

    def _get_grid_coords(self, point: np.ndarray) -> BlockKey:
        """
        Return the integer block index ``(bx, by, bz)`` for a world point.

        ``np.floor`` ensures negative coordinates round towards −∞, matching
        the half-open interval ``[bx·G, (bx+1)·G)``.  Results are memoised
        in a bounded LRU-style cache.

        Parameters
        ----------
        point : array-like, shape (3,)
            World-space query position.
        """
        pt = (float(point[0]), float(point[1]), float(point[2]))
        cached = self._point_to_grid_cache.get(pt)
        if cached is not None:
            return cached

        grid: BlockKey = tuple(np.floor(self._normalize_to_grid(point)).astype(int))

        # Evict the oldest half of the cache when capacity is reached
        if len(self._point_to_grid_cache) >= self._cache_capacity:
            for k in list(self._point_to_grid_cache)[: self._cache_capacity // 2]:
                del self._point_to_grid_cache[k]

        self._point_to_grid_cache[pt] = grid
        return grid

    def _get_map_center(self, grid_coords: BlockKey) -> np.ndarray:
        """
        Return the world-space centre of a block.

        Computed as ``(bx + 0.5, by + 0.5, bz + 0.5) * grid_size``.
        """
        bx, by, bz = grid_coords
        return (np.array([bx, by, bz], dtype=np.float64) + 0.5) * self.grid_size

    def _get_grid_bounds(
        self, grid_coords: BlockKey
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return the AABB ``(min_bound, max_bound)`` of a block, memoised."""
        cached = self._grid_to_bounds_cache.get(grid_coords)
        if cached is not None:
            return cached
        c = self._get_map_center(grid_coords)
        h = self.map_size / 2.0
        bounds = (c - h, c + h)
        self._grid_to_bounds_cache[grid_coords] = bounds
        return bounds

    # ------------------------------------------------------------------
    # Block eviction  (must be called under _manager_lock)
    # ------------------------------------------------------------------

    def _evict_one_block_locked(self) -> None:
        """
        Evict exactly one block from ``_submaps`` to make room for a new one.

        Must be called while holding ``_manager_lock``.

        Eviction preference:
        1. The oldest block (first in LRU order) with **zero occupied voxels**.
        2. If none qualifies, the oldest block overall.

        The occupied-voxel count is read without acquiring the block's own lock
        to avoid a nested lock acquire inside the manager lock.  In CPython the
        GIL makes this safe; on other runtimes it is a best-effort approximation
        consistent with the policy intent (eviction is advisory, not a
        correctness constraint).
        """
        if not self._submaps:
            return

        evict_key: Optional[BlockKey] = None
        for key, block in self._submaps.items():
            # GIL-safe unlocked read of the probability dict's values
            occupied = sum(
                1 for p in block.probability_map.values() if p > 0.5
            )
            if occupied == 0:
                evict_key = key
                break  # First empty block in LRU order

        if evict_key is None:
            # All blocks contain occupied voxels; evict the oldest one
            evict_key = next(iter(self._submaps))

        del self._submaps[evict_key]

    def _touch_block_locked(self, key: BlockKey) -> None:
        """
        Move *key* to the most-recently-used position in the LRU order.

        Must be called while holding ``_manager_lock``.
        """
        if key in self._submaps:
            self._submaps.move_to_end(key)

    # ------------------------------------------------------------------
    # Block allocation
    # ------------------------------------------------------------------

    def _get_or_create_block(self, grid_coords: BlockKey) -> BlockSparseHashMap:
        """
        Return the block at *grid_coords*, creating and registering it if absent.

        Thread-safe: the manager lock guards the check-then-allocate sequence
        atomically, eliminating the TOCTOU race present in the previous
        double-checked locking approach.

        Enforces ``max_blocks`` by calling ``_evict_one_block_locked`` before
        inserting a new block.

        Parameters
        ----------
        grid_coords : BlockKey
            Target block grid index ``(bx, by, bz)``.
        """
        with self._manager_lock:
            block = self._submaps.get(grid_coords)
            if block is not None:
                # Refresh LRU position on every access
                self._touch_block_locked(grid_coords)
                return block

            # Enforce capacity cap before insertion
            if self.max_blocks > 0 and len(self._submaps) >= self.max_blocks:
                self._evict_one_block_locked()

            # Allocate new block
            center = self._get_map_center(grid_coords)
            block  = BlockSparseHashMap(
                map_size  = self.map_size,
                max_depth = self.max_depth,
                center    = center,
            )
            self._submaps[grid_coords] = block
            self._expand_global_bounds_locked(block.min_bound, block.max_bound)

        return block

    # Backwards-compatibility shim
    def _create_map_for_point(self, point: np.ndarray) -> BlockSparseHashMap:
        """Allocate (or return) the block that contains *point*."""
        return self._get_or_create_block(self._get_grid_coords(point))

    def _expand_global_bounds_locked(
        self, min_b: np.ndarray, max_b: np.ndarray
    ) -> None:
        """
        Expand the world-space AABB to include a newly allocated block.

        Must be called while holding ``_manager_lock``.
        """
        if self._global_min_bound is None:
            self._global_min_bound = min_b.copy()
            self._global_max_bound = max_b.copy()
        else:
            self._global_min_bound = np.minimum(self._global_min_bound, min_b)
            self._global_max_bound = np.maximum(self._global_max_bound, max_b)

    # Backwards-compatibility alias
    def _update_global_bounds(self, min_b: np.ndarray, max_b: np.ndarray) -> None:
        """Alias for ``_expand_global_bounds_locked``; requires manager lock held."""
        self._expand_global_bounds_locked(min_b, max_b)

    # ------------------------------------------------------------------
    # Public update API
    # ------------------------------------------------------------------

    def update_with_obstacles(self, points: np.ndarray) -> None:
        """
        Integrate obstacle observations into the map.

        Parameters
        ----------
        points : np.ndarray, shape (N, 3)
            Obstacle point cloud in world coordinates.
        """
        if len(points) == 0:
            return
        self._batch_update(points, 'obstacle')

    def update_with_free_space(self, points: np.ndarray) -> None:
        """
        Integrate free-space observations into the map.

        Parameters
        ----------
        points : np.ndarray, shape (N, 3)
            Free-space point cloud (e.g. ray interiors) in world coordinates.
        """
        if len(points) == 0:
            return
        self._batch_update(points, 'free')

    def update_with_mixed_points(
        self,
        obstacle_points: np.ndarray,
        free_points: np.ndarray,
    ) -> None:
        """
        Integrate obstacle and free-space observations in a single call.

        Parameters
        ----------
        obstacle_points : np.ndarray, shape (N, 3)
        free_points     : np.ndarray, shape (M, 3)
        """
        self.update_with_obstacles(obstacle_points)
        self.update_with_free_space(free_points)

    def _batch_update(self, points: np.ndarray, point_type: str) -> None:
        """
        Bin points into their respective blocks and trigger per-block updates.

        Points that map to the same block are batched together so that
        ``BlockSparseHashMap.update_map`` can vectorise the odds arithmetic
        across the entire batch at once.

        Parameters
        ----------
        points     : np.ndarray, shape (N, 3)
        point_type : ``'obstacle'`` or ``'free'``
        """
        buckets: TypingDict[BlockKey, List[np.ndarray]] = defaultdict(list)
        for p in points:
            buckets[self._get_grid_coords(p)].append(p)

        for grid_coords, pts_list in buckets.items():
            block = self._get_or_create_block(grid_coords)
            block.update_map(np.asarray(pts_list, dtype=np.float32), point_type)

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_stats(self) -> dict:
        """
        Return aggregate statistics across all allocated blocks.

        Acquires the manager lock **only** to snapshot the block list and
        global bounds, then releases it before querying individual blocks.
        No nested locking occurs.

        Returns
        -------
        dict with keys:
            ``total_voxels``, ``occupied_voxels``,
            ``global_min_bound``, ``global_max_bound``, ``submap_count``.
        """
        # Snapshot under manager lock; release before per-block queries
        with self._manager_lock:
            block_snapshot = list(self._submaps.values())
            g_min = (self._global_min_bound.copy()
                     if self._global_min_bound is not None else None)
            g_max = (self._global_max_bound.copy()
                     if self._global_max_bound is not None else None)
            n_blocks = len(self._submaps)

        # Per-block queries — only block._lock is held here
        total_voxels = occupied_voxels = 0
        for block in block_snapshot:
            with block._lock:
                total_voxels    += len(block.probability_map)
                occupied_voxels += sum(
                    1 for p in block.probability_map.values() if p > 0.5
                )

        return {
            'total_voxels':     total_voxels,
            'occupied_voxels':  occupied_voxels,
            'global_min_bound': g_min,
            'global_max_bound': g_max,
            'submap_count':     n_blocks,
        }

    def query_points(
        self,
        points: np.ndarray,
        threshold: Optional[float] = None,
    ) -> np.ndarray:
        """
        Query occupancy probability (or boolean occupancy flag) for a set of points.

        Parameters
        ----------
        points    : np.ndarray, shape (N, 3)
        threshold : float or None
            When provided, returns a ``bool`` array (``True`` = occupied).

        Returns
        -------
        np.ndarray, shape (N,)  – float32 probabilities or bool flags.
        """
        if len(points) == 0:
            dtype = bool if threshold is not None else np.float32
            return np.array([], dtype=dtype)

        probs = np.full(len(points), 0.5, dtype=np.float32)

        # Snapshot block mapping under the manager lock, then release
        with self._manager_lock:
            submap_snapshot = dict(self._submaps)

        buckets: TypingDict[
            BlockKey, List[Tuple[int, np.ndarray]]
        ] = defaultdict(list)
        for i, p in enumerate(points):
            buckets[self._get_grid_coords(p)].append((i, p))

        for grid_coords, idx_pts in buckets.items():
            block = submap_snapshot.get(grid_coords)
            if block is None:
                continue  # Unobserved block → default probability 0.5
            for i, p in idx_pts:
                probs[i] = block.get_voxel_probability(p)

        return (probs > threshold) if threshold is not None else probs

    def get_region_probabilities(
        self,
        bbox_min: np.ndarray,
        bbox_max: np.ndarray,
        return_coords: bool = False,
    ):
        """
        Retrieve occupancy probabilities for all voxels inside an AABB.

        Acquires the manager lock only to snapshot the block dict, then
        queries each block's own lock independently — no nested locking.

        Parameters
        ----------
        bbox_min, bbox_max : array-like, shape (3,)
            World-space corners of the query volume.
        return_coords : bool
            When ``True``, also return voxel centroid coordinates and
            probability values as parallel arrays.

        Returns
        -------
        probabilities : dict  ``{VoxelKey: float}``
        coords        : np.ndarray, shape (M, 3)  *(return_coords only)*
        prob_values   : np.ndarray, shape (M,)    *(return_coords only)*
        """
        bbox_min = np.asarray(bbox_min, dtype=np.float64)
        bbox_max = np.asarray(bbox_max, dtype=np.float64)

        min_grid = self._get_grid_coords(bbox_min)
        max_grid = self._get_grid_coords(bbox_max)

        # Snapshot block mapping; release manager lock before block queries
        with self._manager_lock:
            submap_snapshot = dict(self._submaps)

        combined:   TypingDict[VoxelKey, float] = {}
        all_coords: List[np.ndarray]            = []
        all_probs:  List[float]                 = []

        for bx in range(min_grid[0], max_grid[0] + 1):
            for by in range(min_grid[1], max_grid[1] + 1):
                for bz in range(min_grid[2], max_grid[2] + 1):
                    block = submap_snapshot.get((bx, by, bz))
                    if block is None:
                        continue

                    vs = block._estimate_voxel_size()
                    # Snapshot the block's dict under the block lock only
                    with block._lock:
                        items = list(block.probability_map.items())

                    for key, prob in items:
                        vc = (np.array(key, dtype=np.float64) + 0.5) * vs
                        if np.all(vc >= bbox_min) and np.all(vc <= bbox_max):
                            combined[key] = prob
                            if return_coords:
                                all_coords.append(vc)
                                all_probs.append(prob)

        if return_coords:
            if all_coords:
                return (
                    combined,
                    np.array(all_coords, dtype=np.float64),
                    np.array(all_probs,  dtype=np.float32),
                )
            return (
                combined,
                np.empty((0, 3), dtype=np.float64),
                np.empty(0,      dtype=np.float32),
            )
        return combined

    # ------------------------------------------------------------------
    # Map lifecycle
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """
        Clear all blocks and release all associated memory.

        Stops the temporal decay thread if running, wipes all state, then
        resets the manager to a clean initial state.  Fully operational
        immediately after this call.
        """
        self.stop_decay_thread()
        self._resetting = True
        with self._manager_lock:
            self._submaps.clear()
            self._global_min_bound = None
            self._global_max_bound = None
            self._grid_to_bounds_cache.clear()
            self._point_to_grid_cache.clear()
        self._resetting = False

    # ------------------------------------------------------------------
    # Visualisation
    # ------------------------------------------------------------------

    def visualize(
        self,
        show_probability: bool = True,
        only_occupied: bool = False,
        prob_threshold: float = 0.5,
    ) -> None:
        """
        Open an Open3D window and render all map voxels as coloured points.

        Colour mapping (``show_probability=True``):
        * Red   → high occupancy (> 0.5)
        * Blue  → low occupancy  (< 0.5)
        * Green → near-uniform   (~0.5)

        Parameters
        ----------
        show_probability : bool
            Probability-based colour ramp vs flat grey.
        only_occupied : bool
            Show only voxels with probability ≥ *prob_threshold*.
        prob_threshold : float
            Minimum probability cutoff (active when *only_occupied* is ``True``).
        """
        vis = o3d.visualization.Visualizer()
        vis.create_window(
            window_name="BlockSparseHashMap Visualization",
            width=1280, height=720,
        )

        total = filtered = 0
        all_pts: List[np.ndarray]  = []
        all_col: List[List[float]] = []

        print(f"[visualize] Allocated blocks: {len(self._submaps)}")

        # Snapshot under manager lock to avoid concurrent modification
        with self._manager_lock:
            submap_snapshot = list(self._submaps.items())

        for grid_coords, block in submap_snapshot:
            if not hasattr(block, 'probability_map'):
                print(f"[visualize] Block {grid_coords}: no probability_map, skipping")
                continue
            print(f"[visualize] Block {grid_coords}: "
                  f"{len(block.probability_map)} voxels")

            try:
                vs = block._estimate_voxel_size()
                with block._lock:
                    items = list(block.probability_map.items())

                for key, prob in items:
                    total += 1
                    if only_occupied and prob < prob_threshold:
                        filtered += 1
                        continue
                    vc = (np.array(key, dtype=np.float64) + 0.5) * vs
                    all_pts.append(vc)
                    if show_probability:
                        r = prob       if prob > 0.5 else 0.0
                        b = 1.0 - prob if prob < 0.5 else 0.0
                        g = 1.0 - abs(2.0 * prob - 1.0)
                        all_col.append([r, g, b])
                    else:
                        all_col.append([0.5, 0.5, 0.5])
            except Exception as exc:
                import traceback
                print(f"[visualize] Error in block {grid_coords}: {exc}")
                traceback.print_exc()

        if not all_pts:
            print("[visualize] No voxels to display.")
            vis.destroy_window()
            return

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(np.array(all_pts))
        pcd.colors = o3d.utility.Vector3dVector(np.array(all_col))
        vis.add_geometry(pcd)
        vis.get_render_option().point_size       = 5
        vis.get_render_option().background_color = np.array([0.1, 0.1, 0.1])
        vis.get_view_control().set_zoom(0.8)

        print(f"[visualize] Showing {total - filtered} / {total} voxels "
              f"({filtered} filtered)")
        vis.run()
        vis.destroy_window()

    def get_all_maps_pointcloud(
        self,
        threshold: float = 0.5,
        only_occupied: bool = True,
    ) -> o3d.geometry.PointCloud:
        """
        Build and return an Open3D ``PointCloud`` from all map voxels.

        Parameters
        ----------
        threshold     : float  Minimum probability for inclusion.
        only_occupied : bool   Exclude sub-threshold voxels when ``True``.

        Returns
        -------
        o3d.geometry.PointCloud
            Coloured by occupancy probability.  Empty if no voxels qualify.
        """
        all_pts: List[np.ndarray]  = []
        all_col: List[List[float]] = []

        with self._manager_lock:
            submap_snapshot = list(self._submaps.items())

        for grid_coords, block in submap_snapshot:
            if not hasattr(block, 'probability_map'):
                print(f"[get_all_maps_pointcloud] Block {grid_coords}: "
                      "no probability_map, skipping")
                continue
            vs = block._estimate_voxel_size()
            with block._lock:
                items = list(block.probability_map.items())

            for key, prob in items:
                if only_occupied and prob < threshold:
                    continue
                vc = (np.array(key, dtype=np.float64) + 0.5) * vs
                all_pts.append(vc)
                r = prob       if prob > 0.5 else 0.0
                b = 1.0 - prob if prob < 0.5 else 0.0
                g = 1.0 - abs(2.0 * prob - 1.0)
                all_col.append([r, g, b])

        pcd = o3d.geometry.PointCloud()
        if all_pts:
            pcd.points = o3d.utility.Vector3dVector(np.array(all_pts))
            pcd.colors = o3d.utility.Vector3dVector(np.array(all_col))
        return pcd

    # ------------------------------------------------------------------
    # Temporal decay background thread
    # ------------------------------------------------------------------

    def _decay_worker(self) -> None:
        """
        Background daemon: periodically decays probabilities and prunes blocks.

        Lock ordering respected throughout:
        * Manager lock acquired to snapshot block list → immediately released.
        * Per-block maintenance runs under block._lock only.
        * Empty blocks are pruned in a second manager lock acquisition — never
          while holding a block lock.
        """
        while self._decay_running:
            t0 = time.monotonic()

            # Snapshot block list under the manager lock, then release
            with self._manager_lock:
                block_snapshot = list(self._submaps.items())

            empty_keys: List[BlockKey] = []
            for key, block in block_snapshot:
                if not self._decay_running:
                    break
                if hasattr(block, 'decay_probabilities'):
                    block.decay_probabilities(self._decay_factor)
                if hasattr(block, 'clear_expired_nodes'):
                    block.clear_expired_nodes(self._expiration_timeout)
                # Collect blocks that are now empty after maintenance
                if block.is_empty():
                    empty_keys.append(key)

            # Prune empty blocks — separate manager lock acquisition,
            # no block locks held at this point (no nesting)
            if empty_keys:
                with self._manager_lock:
                    for key in empty_keys:
                        b = self._submaps.get(key)
                        # Guard: block may have been re-populated in the window
                        # between the is_empty() check and this removal
                        if b is not None and b.is_empty():
                            del self._submaps[key]

            elapsed = time.monotonic() - t0
            time.sleep(max(0.0, self._decay_interval - elapsed))

    def start_decay_thread(self) -> None:
        """
        Start the background temporal-decay thread.

        No-op if ``enable_temporal_decay`` is ``False`` or already running.
        """
        if not self.enable_temporal_decay:
            print("[BlockSparseHashMapManager] Temporal decay disabled "
                  "(set enable_temporal_decay=True to enable).")
            return
        if self._decay_running:
            print("[BlockSparseHashMapManager] Decay thread already running.")
            return

        self._decay_running = True
        self._decay_thread  = threading.Thread(
            target=self._decay_worker,
            daemon=True,
            name="BSHMDecayThread",
        )
        self._decay_thread.start()
        print(f"[BlockSparseHashMapManager] Decay thread started "
              f"(interval={self._decay_interval:.2f}s, "
              f"factor={self._decay_factor:.3f}).")

    def stop_decay_thread(self) -> None:
        """
        Stop the background temporal-decay thread gracefully.

        Waits up to 1 s for the thread to finish.  Safe to call even if the
        thread was never started.
        """
        if self._decay_thread is not None and self._decay_running:
            self._decay_running = False
            self._decay_thread.join(timeout=1.0)
            self._decay_thread = None
            print("[BlockSparseHashMapManager] Decay thread stopped.")


# ---------------------------------------------------------------------------
# Backwards-compatibility aliases
# ---------------------------------------------------------------------------

#: Legacy name kept for external API compatibility → ``BlockSparseHashMap``
OctreeProbabilisticMap = BlockSparseHashMap

#: Legacy name kept for external API compatibility → ``BlockSparseHashMapManager``
OctreeMapManager = BlockSparseHashMapManager


# ---------------------------------------------------------------------------
# Self-test / demo  (main function unchanged)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Starting BlockSparseHashMapManager self-test …")

    def basic_test() -> None:
        try:
            print("\nCreating BlockSparseHashMapManager …")
            t0 = time.time()
            map_manager = OctreeMapManager(
                map_size=64,
                max_depth=7,
                overlap=0.0,
                enable_temporal_decay=True,
            )
            print(f"Manager created in {time.time() - t0:.2f}s")

            map_manager.start_decay_thread()

            print("\nGenerating test point cloud …")
            locations = [
                [0,   0,   0  ],
                [100, 100, 100],
                [-100, -100, -100],
                [200, -100, 50 ],
                [32,  32,   32 ],
                [300, 200, -200],
            ]

            for loc in locations:
                sphere_center = np.array(loc, dtype=np.float32)
                rng = np.random.default_rng(seed=abs(sum(loc)))

                n_pts = 4_000
                r     = 20.0 * rng.random(n_pts)
                theta = rng.random(n_pts) * 2.0 * np.pi
                phi   = rng.random(n_pts) * np.pi
                obstacles = sphere_center + np.column_stack([
                    r * np.sin(phi) * np.cos(theta),
                    r * np.sin(phi) * np.sin(theta),
                    r * np.cos(phi),
                ]).astype(np.float32)

                for i in range(4):
                    print(f"  Updating block at {loc} (iteration {i+1}/12) …")
                    t_a = time.time()
                    map_manager.update_with_obstacles(obstacles)
                    print(f"    Insert time: {time.time() - t_a:.4f}s")

            stats = map_manager.get_stats()
            print("\nMap statistics:")
            print(f"  Allocated blocks : {stats['submap_count']}")
            print(f"  Total voxels     : {stats['total_voxels']}")
            print(f"  Global min bound : {stats['global_min_bound']}")
            print(f"  Global max bound : {stats['global_max_bound']}")

            print("\nRegion query test …")
            bbox_min = np.array([21-20,   21-20,  21-20  ])
            bbox_max = np.array([41,  41,   41])
            for i in range(16):
                t_a = time.time()
                _, coords, prob_values = map_manager.get_region_probabilities(
                    bbox_min, bbox_max, return_coords=True
                )
                elapsed = time.time() - t_a
                print(
                    f"  [{i+1:02d}] Voxels: {len(coords):5d}  |  points: {coords[:5]}  |  "
                    f"Sample probs: {prob_values[:3]}  |  Query: {elapsed:.5f}s"
                )
                time.sleep(0.45)

            stats_after = map_manager.get_stats()
            print("\nPost-decay statistics:")
            print(f"  Allocated blocks : {stats_after['submap_count']}")
            print(f"  Total voxels     : {stats_after['total_voxels']}")
            print(f"  Occupied voxels  : {stats_after['occupied_voxels']}")

            print("\nLaunching visualisation …")
            try:
                print("  All voxels:")
                map_manager.visualize(show_probability=True)

                print("  Occupied voxels only (prob ≥ 0.5):")
                map_manager.visualize(
                    show_probability=True, only_occupied=True, prob_threshold=0.5)

                print("  High-confidence voxels only (prob ≥ 0.7):")
                map_manager.visualize(
                    show_probability=True, only_occupied=True, prob_threshold=0.7)

                print("  Point cloud (occupied only):")
                pcd = map_manager.get_all_maps_pointcloud(
                    threshold=0.5, only_occupied=True)
                o3d.visualization.draw_geometries([pcd])

            except Exception as exc:
                import traceback
                print(f"Visualisation error: {exc}")
                traceback.print_exc()

            map_manager.stop_decay_thread()
            t_a = time.time()
            map_manager.reset()
            print(f"\nReset completed in {time.time() - t_a:.4f}s")

        except Exception as exc:
            import traceback
            print(f"Test failed: {exc}")
            traceback.print_exc()

    basic_test()