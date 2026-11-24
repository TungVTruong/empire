"""
Procedural Starship stack generator for Blender (4.2 LTS - 5.x).

Builds a two-stage, Starship-class stack: a ship with tiled heat shield,
hinged forward/aft flaps and a hinged payload door, plus a booster with
a vented hot-staging ring, rotating grid fins and a 33-engine cluster.
Every hinge is a real pivot (mount empty -> animated hinge empty -> panel),
so you can pose or re-key it by hand afterwards.

Layout
------
    Parameters ........ dataclasses at the top; every major dimension is exposed
    Core / caching .... hashing, ensure_* helpers, garbage collection
    Mesh builders ..... pure bmesh functions (no bpy.ops, safe headless)
    Materials ......... build_materials()
    Geometry .......... build_geometry()  -> returns a Rig of hinges
    Camera / world .... setup_camera(), setup_environment(), configure_render()
    Animation ......... animate()
    Reporting ......... RunReport (timings), report_ai_usage()
    Entry points ...... generate() for Python callers, main() for the CLI

Incremental rebuilds
--------------------
Each mesh, material, the world and the animation store a hash of exactly the
parameters they depend on (plus GENERATOR_VERSION). Re-running the script with
unchanged parameters reuses those datablocks; changing, say, only the camera
rebuilds only the camera. Objects that are no longer needed (e.g. you reduced
grid fins from 4 to 3) are removed. Pass --force to rebuild everything.

Run
---
    blender -b --factory-startup --python starship_generator.py -- --remove-default-cube --save starship.blend
    blender -b --python starship_generator.py -- --set ship.length=50 --set grid_fins.count=3 --render out.png
    python starship_generator.py -- --dump-params     # with `pip install bpy`

Inside Blender: open in the Text Editor and press Run Script (defaults apply).

Accuracy note: dimensions are public, approximate figures for a Block 2-era
vehicle and are exposed as parameters precisely because they change between
vehicle versions. No logos or trademarks are modelled.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import sys
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import bpy  # must precede bmesh/mathutils when run as the `bpy` pip module
import bmesh
from mathutils import Matrix, Vector

GENERATOR_VERSION = "1.0.1"
TAG = "starship_gen"          # custom property marking IDs this script owns
HASH_KEY = "starship_gen_hash"
PREFIX = "SS_"
EPS = 1e-6

log = logging.getLogger("starship_gen")


# =============================================================================
# Parameters
# =============================================================================

@dataclass
class HullParams:
    radius: float = 4.5             # 9 m diameter stainless hull
    segments: int = 96              # radial resolution of every hull lathe
    ring_height: float = 1.83       # pitch of the welded barrel rings (approx.)


@dataclass
class ShipParams:
    length: float = 52.1            # base of skirt to nose tip (Block 2-ish, approx.)
    nose_length: float = 18.0       # tangent-ogive nose section
    nose_tip_radius: float = 0.45   # blunting radius at the tip
    nose_samples: int = 28
    tile_arc_deg: float = 190.0     # windward heat-shield coverage, centred on -Y
    engine_mount_z: float = 4.3     # gimbal plane above skirt base
    rsl_count: int = 3              # sea-level engines (centre)
    rsl_ring_radius: float = 1.0
    rvac_count: int = 3             # vacuum engines (outer)
    rvac_ring_radius: float = 3.05
    raceway: bool = True
    raceway_azimuth_deg: float = 70.0
    raceway_width: float = 0.6
    raceway_depth: float = 0.3


@dataclass
class BoosterParams:
    length: float = 71.0            # includes the hot-staging ring
    hot_stage_height: float = 1.8
    hot_stage_struts: int = 30
    hot_stage_strut_width: float = 0.35
    hot_stage_band: float = 0.3
    engine_exit_z: float = 0.25     # nozzle exit height above booster base
    # (count, ring radius) from centre outwards: 3 + 10 + 20 = 33 engines
    engine_rings: Tuple[Tuple[int, float], ...] = ((3, 0.95), (10, 2.45), (20, 3.88))
    raceway: bool = True
    raceway_azimuth_deg: float = 90.0
    raceway_width: float = 0.7
    raceway_depth: float = 0.35


@dataclass
class EngineParams:
    segments: int = 40
    sl_length: float = 3.1          # gimbal to nozzle exit
    sl_exit_radius: float = 0.6
    vac_length: float = 4.6
    vac_exit_radius: float = 1.15
    throat_radius: float = 0.2
    powerhead_radius: float = 0.55
    powerhead_height: float = 1.0
    wall_thickness: float = 0.03


@dataclass
class FlapParams:
    enabled: bool = True
    thickness: float = 0.3
    fairing_radius: float = 0.42    # hinge aerocover
    stowed_angle_deg: float = 70.0  # folded toward leeward (+Y)
    deployed_angle_deg: float = 0.0 # 0 = spread radially
    forward_hinge_z: float = 37.5   # bottom of forward hinge, from ship base
    forward_root_chord: float = 7.0
    forward_tip_chord: float = 3.2
    forward_span: float = 3.4
    forward_tip_shift: float = 0.75 # 0 = tip aligned to hinge bottom, 1 = to top
    forward_windward_offset_deg: float = -20.0  # negative = moved leeward
    aft_hinge_z: float = 1.2
    aft_root_chord: float = 10.0
    aft_tip_chord: float = 6.0
    aft_span: float = 4.0
    aft_tip_shift: float = 0.0
    aft_windward_offset_deg: float = 15.0


@dataclass
class GridFinParams:
    enabled: bool = True
    azimuths_deg: Tuple[float, ...] = (40.0, 140.0, 220.0, 320.0)
    span: float = 3.6               # radial extent
    width: float = 2.4
    chord: float = 0.5              # depth along the flow
    cells_span: int = 8
    cells_width: int = 5
    frame_thickness: float = 0.12
    web_thickness: float = 0.035
    root_gap: float = 0.6           # shaft length between hull and fin
    shaft_radius: float = 0.14
    z_below_tank_top: float = 2.2


@dataclass
class PayloadDoorParams:
    enabled: bool = True
    azimuth_deg: float = 90.0       # leeward
    top_z: float = 32.0             # hinge line, from ship base
    width: float = 5.0
    height: float = 1.2
    thickness: float = 0.08


@dataclass
class MaterialParams:
    steel_color: Tuple[float, float, float] = (0.80, 0.80, 0.82)
    steel_roughness: float = 0.2
    steel_anisotropy: float = 0.55
    steel_streak_scale: float = 6.0
    weld_seam_width: float = 0.035
    weld_bump_strength: float = 0.12
    booster_soot: float = 0.55
    booster_soot_height: float = 22.0
    tile_color: Tuple[float, float, float] = (0.022, 0.022, 0.025)
    tile_roughness: float = 0.72
    tile_size: float = 0.4          # Voronoi cell size, metres (stylised)
    grout_width: float = 0.025
    bell_roughness: float = 0.33


@dataclass
class AnimationParams:
    enabled: bool = True
    fps: int = 24
    frame_start: int = 1
    frame_end: int = 240
    flap_deploy_start: int = 20
    flap_deploy_end: int = 70
    flap_flutter_deg: float = 9.0   # differential control motion after deploy
    flap_flutter_interval: int = 14
    gridfin_sweep_deg: float = 18.0
    gridfin_period: int = 60
    door_open_start: int = 90
    door_open_end: int = 130
    door_open_deg: float = 95.0
    separation_start: int = 160
    separation_end: int = 235
    separation_distance: float = 25.0  # 0 disables stage separation
    camera_orbit_deg: float = 110.0


@dataclass
class CameraParams:
    lens_mm: float = 50.0
    distance: float = 185.0
    height: float = 30.0
    target_z: float = 62.0
    start_azimuth_deg: float = -60.0  # -90 looks straight at the heat shield
    clip_end: float = 5000.0
    use_dof: bool = False
    f_stop: float = 8.0


@dataclass
class EnvironmentParams:
    sun_elevation_deg: float = 25.0
    sun_azimuth_deg: float = -40.0
    sun_strength: float = 4.0
    sun_angle_deg: float = 0.53
    sky_strength: float = 0.35
    fallback_color: Tuple[float, float, float] = (0.35, 0.45, 0.6)


@dataclass
class RenderParams:
    engine: str = "CYCLES"          # CYCLES | EEVEE
    samples: int = 64
    resolution_x: int = 1080
    resolution_y: int = 1920
    resolution_percentage: int = 50
    denoise: bool = True


@dataclass
class StarshipParams:
    include_booster: bool = True
    hull: HullParams = field(default_factory=HullParams)
    ship: ShipParams = field(default_factory=ShipParams)
    booster: BoosterParams = field(default_factory=BoosterParams)
    engines: EngineParams = field(default_factory=EngineParams)
    flaps: FlapParams = field(default_factory=FlapParams)
    grid_fins: GridFinParams = field(default_factory=GridFinParams)
    payload_door: PayloadDoorParams = field(default_factory=PayloadDoorParams)
    materials: MaterialParams = field(default_factory=MaterialParams)
    animation: AnimationParams = field(default_factory=AnimationParams)
    camera: CameraParams = field(default_factory=CameraParams)
    environment: EnvironmentParams = field(default_factory=EnvironmentParams)
    render: RenderParams = field(default_factory=RenderParams)


def _coerce(current, value, path):
    """Coerce a JSON value onto the type of the existing default."""
    if isinstance(current, bool):
        if not isinstance(value, bool):
            raise TypeError(f"{path}: expected true/false, got {value!r}")
        return value
    if isinstance(current, int) and not isinstance(current, bool):
        if isinstance(value, float) and not value.is_integer():
            raise TypeError(f"{path}: expected an integer, got {value!r}")
        return int(value)
    if isinstance(current, float):
        return float(value)
    if isinstance(current, tuple):
        def deep_tuple(v):
            return tuple(deep_tuple(x) for x in v) if isinstance(v, (list, tuple)) else v
        if not isinstance(value, (list, tuple)):
            raise TypeError(f"{path}: expected a list, got {value!r}")
        return deep_tuple(value)
    return type(current)(value)


def apply_overrides(target, data: dict, path: str = "") -> None:
    """Deep-merge a dict into a params dataclass, rejecting unknown keys."""
    valid = {f.name for f in fields(target)}
    for key, value in data.items():
        full = f"{path}{key}"
        if key not in valid:
            raise KeyError(f"Unknown parameter '{full}'. Valid: {sorted(valid)}")
        current = getattr(target, key)
        if is_dataclass(current):
            if not isinstance(value, dict):
                raise TypeError(f"{full}: expected an object")
            apply_overrides(current, value, full + ".")
        else:
            setattr(target, key, _coerce(current, value, full))


def set_param(params: StarshipParams, dotted: str, raw: str) -> None:
    """Apply one `section.key=value` override (value parsed as JSON if possible)."""
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        value = raw
    node: dict = {}
    cursor = node
    parts = dotted.split(".")
    for p in parts[:-1]:
        cursor = cursor.setdefault(p, {})
    cursor[parts[-1]] = value
    apply_overrides(params, node)


def validate(P: StarshipParams) -> None:
    errs = []
    R = P.hull.radius
    nose_base = P.ship.length - P.ship.nose_length
    if P.ship.nose_length <= R:
        errs.append("ship.nose_length must exceed hull.radius for a tangent ogive")
    if nose_base <= P.ship.engine_mount_z:
        errs.append("ship.nose_length leaves no barrel above the engine mount")
    if P.flaps.enabled:
        fz0 = P.flaps.forward_hinge_z
        if not (nose_base <= fz0 and fz0 + P.flaps.forward_root_chord < P.ship.length - 2.0):
            errs.append("flaps.forward_hinge_z must place the forward flap hinge on the nose cone")
        if P.flaps.aft_hinge_z + P.flaps.aft_root_chord > nose_base:
            errs.append("aft flap extends onto the nose cone")
    if P.payload_door.enabled:
        d = P.payload_door
        if not (P.ship.engine_mount_z < d.top_z - d.height and d.top_z < nose_base):
            errs.append("payload_door must sit on the cylindrical barrel")
        if d.width / R >= math.pi:
            errs.append("payload_door.width is wider than half the hull")
    if P.include_booster:
        for count, radius in P.booster.engine_rings:
            if count < 1 or radius + P.engines.sl_exit_radius + P.engines.wall_thickness / 2 > R:
                errs.append(f"booster engine ring ({count}, {radius}) does not fit inside the hull")
            if count > 1 and 2 * math.pi * radius / count < 2 * P.engines.sl_exit_radius - EPS and radius > EPS:
                errs.append(f"booster engine ring ({count}, {radius}): nozzles would overlap")
        if P.booster.hot_stage_height >= P.booster.length:
            errs.append("booster.hot_stage_height exceeds booster.length")
    if P.ship.rvac_ring_radius + P.engines.vac_exit_radius > R:
        errs.append("vacuum engines do not fit inside the ship skirt")
    a = P.animation
    if a.enabled and not (a.frame_start < a.frame_end):
        errs.append("animation.frame_start must be before frame_end")
    if P.render.engine.upper() not in ("CYCLES", "EEVEE"):
        errs.append("render.engine must be CYCLES or EEVEE")
    if errs:
        raise ValueError("Invalid parameters:\n  - " + "\n  - ".join(errs))


# =============================================================================
# Core: hashing, caching, object management
# =============================================================================

def param_hash(*parts) -> str:
    payload = [GENERATOR_VERSION] + [asdict(p) if is_dataclass(p) else p for p in parts]
    blob = json.dumps(payload, sort_keys=True, default=list).encode()
    return hashlib.sha1(blob).hexdigest()[:16]


def geometry_fields(dc, *exclude: str) -> dict:
    """Dataclass as dict minus placement/animation fields, so moving a part never rebuilds its mesh."""
    return {k: v for k, v in asdict(dc).items() if k not in exclude}


class BuildContext:
    """Tracks what was built vs. reused and which generated IDs are still live."""

    def __init__(self, force: bool = False):
        self.force = force
        self.touched_objects: set = set()
        self.touched_meshes: set = set()
        self.counters: Dict[str, int] = {
            "meshes_built": 0, "meshes_reused": 0,
            "materials_built": 0, "materials_reused": 0,
            "objects_created": 0, "objects_removed": 0,
            "animations_built": 0, "animations_reused": 0,
        }

    def bump(self, key: str, n: int = 1) -> None:
        self.counters[key] += n

    def is_fresh(self, idblock, key: str) -> bool:
        return idblock is not None and not self.force and idblock.get(HASH_KEY) == key


def ensure_collection(name: str, parent: Optional[bpy.types.Collection] = None) -> bpy.types.Collection:
    coll = bpy.data.collections.get(name)
    if coll is None:
        coll = bpy.data.collections.new(name)
        coll[TAG] = True
    parent = parent or bpy.context.scene.collection
    if coll.name not in parent.children:
        parent.children.link(coll)
    return coll


def _link_only(obj: bpy.types.Object, coll: bpy.types.Collection) -> None:
    if coll not in obj.users_collection:
        coll.objects.link(obj)
    for other in list(obj.users_collection):
        if other is not coll and other.get(TAG):
            other.objects.unlink(obj)


def _parent(obj, parent) -> None:
    if obj.parent is not parent:
        obj.parent = parent
    obj.matrix_parent_inverse = Matrix.Identity(4)


def ensure_object(ctx: BuildContext, name: str, data, coll, parent=None) -> Tuple[bpy.types.Object, bool]:
    """Get or create an object; returns (object, created). `data=None` gives an empty."""
    obj = bpy.data.objects.get(name)
    want_type = "EMPTY" if data is None else None
    if obj is not None and ((want_type == "EMPTY") != (obj.type == "EMPTY") or
                            (data is not None and type(obj.data) is not type(data))):
        bpy.data.objects.remove(obj, do_unlink=True)
        obj = None
    created = obj is None
    if created:
        obj = bpy.data.objects.new(name, data)
        obj[TAG] = True
        ctx.bump("objects_created")
    elif data is not None and obj.data is not data:
        obj.data = data
    _link_only(obj, coll)
    _parent(obj, parent)
    ctx.touched_objects.add(obj.name)
    return obj, created


def ensure_empty(ctx, name, coll, parent=None, display="PLAIN_AXES", size=1.0):
    obj, created = ensure_object(ctx, name, None, coll, parent)
    obj.empty_display_type = display
    obj.empty_display_size = size
    return obj, created


def ensure_mesh(ctx: BuildContext, name: str, key: str,
                build: Callable[[bmesh.types.BMesh], None],
                materials: Sequence[bpy.types.Material] = (),
                sharp_angle_deg: Optional[float] = 35.0) -> bpy.types.Mesh:
    """Return a mesh datablock, rebuilding it in place only when `key` changed."""
    ctx.touched_meshes.add(name)
    mesh = bpy.data.meshes.get(name)
    if ctx.is_fresh(mesh, key):
        ctx.bump("meshes_reused")
        return mesh
    bm = bmesh.new()
    try:
        build(bm)
        bmesh.ops.recalc_face_normals(bm, faces=bm.faces[:])
        if sharp_angle_deg is not None:
            bm_mark_sharp(bm, sharp_angle_deg)
        if mesh is None:
            mesh = bpy.data.meshes.new(name)
            mesh[TAG] = True
        else:
            mesh.clear_geometry()
        # Slots first: on Blender 4.x/5.x, materials.clear() also drops the
        # per-face material_index attribute, so it must run before to_mesh().
        mesh.materials.clear()
        for mat in materials:
            mesh.materials.append(mat)
        bm.to_mesh(mesh)
    finally:
        bm.free()
    mesh[HASH_KEY] = key
    mesh.update()
    ctx.bump("meshes_built")
    return mesh


def ensure_modifier(obj, name: str, mtype: str, **props):
    mod = obj.modifiers.get(name)
    if mod is None or mod.type != mtype:
        if mod is not None:
            obj.modifiers.remove(mod)
        mod = obj.modifiers.new(name, mtype)
    for k, v in props.items():
        if hasattr(mod, k):
            setattr(mod, k, v)
    return mod


def garbage_collect(ctx: BuildContext) -> None:
    """Remove generated objects/meshes that this run no longer produced."""
    for obj in list(bpy.data.objects):
        if obj.get(TAG) and obj.name not in ctx.touched_objects:
            log.info("Removing stale object %s", obj.name)
            bpy.data.objects.remove(obj, do_unlink=True)
            ctx.bump("objects_removed")
    for mesh in list(bpy.data.meshes):
        if mesh.get(TAG) and mesh.name not in ctx.touched_meshes and mesh.users == 0:
            bpy.data.meshes.remove(mesh)


# =============================================================================
# Mesh builders (pure bmesh)
# =============================================================================

def bm_mark_sharp(bm, angle_deg: float) -> None:
    limit = math.radians(angle_deg)
    for e in bm.edges:
        if len(e.link_faces) == 2 and e.calc_face_angle(0.0) > limit:
            e.smooth = False


def bm_lathe(bm, profile: Sequence[Tuple[float, float]], segments: int, closed: bool = False,
             matrix: Optional[Matrix] = None, mat_fn: Optional[Callable[[Vector], int]] = None,
             mat: int = 0, smooth: bool = True) -> list:
    """Revolve an (r, z) profile around Z. Points with r == 0 collapse to a pole."""
    rings = []
    for r, z in profile:
        if r <= EPS:
            rings.append([bm.verts.new((0.0, 0.0, z))])
        else:
            rings.append([bm.verts.new((r * math.cos(2 * math.pi * i / segments),
                                        r * math.sin(2 * math.pi * i / segments), z))
                          for i in range(segments)])
    pairs = list(zip(rings, rings[1:]))
    if closed:
        pairs.append((rings[-1], rings[0]))
    faces = []
    for a, b in pairs:
        if len(a) == 1 and len(b) == 1:
            continue
        for i in range(segments):
            j = (i + 1) % segments
            if len(a) == 1:
                vs = (a[0], b[j], b[i])
            elif len(b) == 1:
                vs = (a[i], a[j], b[0])
            else:
                vs = (a[i], a[j], b[j], b[i])
            faces.append(bm.faces.new(vs))
    for f in faces:
        f.smooth = smooth
        f.material_index = mat_fn(f.calc_center_median()) if mat_fn else mat
    if matrix is not None:
        bmesh.ops.transform(bm, matrix=matrix, verts=[v for ring in rings for v in ring])
    return faces


def bm_box(bm, center, size, matrix: Optional[Matrix] = None, mat: int = 0) -> set:
    m = Matrix.Translation(Vector(center)) @ Matrix.Diagonal((*size, 1.0))
    if matrix is not None:
        m = matrix @ m
    verts = bmesh.ops.create_cube(bm, size=1.0, matrix=m)["verts"]
    faces = {f for v in verts for f in v.link_faces}
    for f in faces:
        f.material_index = mat
        f.smooth = False
    return faces


def bm_cylinder(bm, radius: float, depth: float, segments: int,
                matrix: Matrix, mat: int = 0, smooth: bool = True) -> set:
    verts = bmesh.ops.create_cone(bm, cap_ends=True, cap_tris=False, segments=segments,
                                  radius1=radius, radius2=radius, depth=depth, matrix=matrix)["verts"]
    faces = {f for v in verts for f in v.link_faces}
    for f in faces:
        f.material_index = mat
        f.smooth = smooth and len(f.verts) == 4
    return faces


def bm_curved_panel(bm, radius: float, half_angle: float, z0: float, z1: float,
                    columns: int, origin: Vector, mat: int = 0) -> None:
    """Single-layer cylindrical patch centred on +X, shifted by -origin (use Solidify)."""
    rows = []
    for z in (z0, z1):
        rows.append([bm.verts.new((radius * math.cos(phi) - origin.x,
                                   radius * math.sin(phi) - origin.y, z - origin.z))
                     for phi in (-half_angle + 2 * half_angle * i / columns for i in range(columns + 1))])
    for i in range(columns):
        f = bm.faces.new((rows[0][i], rows[0][i + 1], rows[1][i + 1], rows[1][i]))
        f.material_index = mat
        f.smooth = True


def ogive_profile(R: float, L: float, z_base: float, tip_radius: float, samples: int):
    """Tangent ogive from (R, z_base) to a blunted tip."""
    rho = (R * R + L * L) / (2 * R)
    pts = []
    for i in range(samples + 1):
        h = L * i / samples
        r = math.sqrt(max(rho * rho - h * h, 0.0)) + R - rho
        if r <= tip_radius and i > 0:
            z_cut = z_base + h
            for k in range(1, 5):
                ang = math.radians(90 * k / 4)
                pts.append((r * math.cos(ang), z_cut + r * 0.7 * math.sin(ang)))
            break
        pts.append((r, z_base + h))
    if pts[-1][0] > EPS:
        pts.append((0.0, pts[-1][1] + tip_radius * 0.5))
    return pts


def ship_profile(P: StarshipParams):
    R, S = P.hull.radius, P.ship
    nose_base = S.length - S.nose_length
    return [(R, 0.0)] + ogive_profile(R, S.nose_length, nose_base, S.nose_tip_radius, S.nose_samples)


def profile_radius_at(profile, z: float) -> float:
    for (r0, z0), (r1, z1) in zip(profile, profile[1:]):
        if z0 <= z <= z1 and z1 > z0:
            return r0 + (r1 - r0) * (z - z0) / (z1 - z0)
    return profile[-1][0] if z > profile[-1][1] else profile[0][0]


def windward_tile_fn(arc_deg: float) -> Callable[[Vector], int]:
    half = arc_deg / 2.0

    def fn(c: Vector) -> int:
        if abs(c.x) < EPS and abs(c.y) < EPS:
            return 1
        az = math.degrees(math.atan2(c.y, c.x))
        diff = abs(((az + 90.0 + 180.0) % 360.0) - 180.0)
        return 1 if diff <= half else 0
    return fn


def build_engine_bm(bm, E: EngineParams, length: float, exit_radius: float) -> None:
    """Raptor-like engine: origin at the gimbal point, nozzle pointing -Z."""
    ph_r, ph_h, rt = E.powerhead_radius, E.powerhead_height, E.throat_radius
    chamber_top = -ph_h - 0.1
    throat_z = chamber_top - 0.55
    profile = [(0.0, 0.05), (ph_r * 0.6, 0.05), (ph_r, -0.15), (ph_r, -ph_h),
               (rt * 1.7, chamber_top), (rt * 1.7, chamber_top - 0.3), (rt, throat_z)]
    bell_len = length + throat_z  # throat_z is negative
    n = 14
    for i in range(1, n + 1):
        s = i / n
        profile.append((rt + (exit_radius - rt) * s ** 0.6, throat_z - bell_len * s))
    bm_lathe(bm, profile, E.segments, mat=0)
    # Turbopumps and manifolds hung off the powerhead for visual asymmetry
    for ang, r, d in ((0.0, 0.22, 0.9), (math.pi, 0.18, 0.8)):
        m = (Matrix.Translation((math.cos(ang) * (ph_r + 0.12), math.sin(ang) * (ph_r + 0.12), -ph_h * 0.5)))
        bm_cylinder(bm, r, d, 16, m, mat=1)
    m = Matrix.Translation((0, 0, -ph_h - 0.05)) @ Matrix.Rotation(math.pi / 2, 4, "Y")  # along the pump axis
    bm_cylinder(bm, 0.08, ph_r * 2.6, 12, m, mat=1)


def build_flap_bm(bm, root: float, tip: float, span: float, shift: float,
                  thickness: float, x0: float, windward_sign: int) -> None:
    """Trapezoidal flap in its hinge frame: hinge along +Z, span along +X."""
    x1 = x0 + span
    tz0 = (root - tip) * shift
    tz1 = tz0 + tip
    t, tt = thickness / 2, thickness * 0.2
    v = [bm.verts.new(p) for p in (
        (x0, -t, 0), (x0, -t, root), (x1, -tt, tz1), (x1, -tt, tz0),
        (x0, t, 0), (x0, t, root), (x1, tt, tz1), (x1, tt, tz0))]
    for idx in ((0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4), (3, 7, 6, 2), (0, 4, 7, 3), (1, 2, 6, 5)):
        bm.faces.new([v[i] for i in idx])
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces[:])
    for f in bm.faces:
        f.smooth = False
        f.material_index = 1 if f.normal.y * windward_sign > 0.5 else 0


def build_grid_fin_bm(bm, G: GridFinParams) -> None:
    """Grid fin in its hinge frame: shaft along +X (radial), lattice in XY, flow along Z."""
    x0, w, c = G.root_gap, G.width, G.chord
    ft, wt = G.frame_thickness, G.web_thickness
    m = Matrix.Translation((x0 / 2 + 0.1, 0, 0)) @ Matrix.Rotation(math.pi / 2, 4, "Y")
    bm_cylinder(bm, G.shaft_radius, x0 + 0.4, 16, m, mat=1)
    xc = x0 + G.span / 2
    for y in (-(w - ft) / 2, (w - ft) / 2):
        bm_box(bm, (xc, y, 0), (G.span, ft, c))
    for x in (x0 + ft / 2, x0 + G.span - ft / 2):
        bm_box(bm, (x, 0, 0), (ft, w, c))
    for i in range(1, G.cells_span):
        bm_box(bm, (x0 + G.span * i / G.cells_span, 0, 0), (wt, w - ft, c * 0.96))
    for j in range(1, G.cells_width):
        bm_box(bm, (xc, -w / 2 + w * j / G.cells_width, 0), (G.span - ft, wt, c * 0.96))


def build_hot_stage_bm(bm, R: float, B: BoosterParams, segments: int) -> None:
    """Vented interstage ring from z=0 to hot_stage_height (booster-local offset applied later)."""
    h, band, t = B.hot_stage_height, B.hot_stage_band, 0.12
    for z0 in (0.0, h - band):
        bm_lathe(bm, [(R, z0), (R, z0 + band), (R - t, z0 + band), (R - t, z0)], segments, closed=True)
    strut_h = h - 2 * band
    for i in range(B.hot_stage_struts):
        a = 2 * math.pi * i / B.hot_stage_struts
        bm_box(bm, (R - t / 2, 0, band + strut_h / 2), (t, B.hot_stage_strut_width, strut_h + 0.02),
               matrix=Matrix.Rotation(a, 4, "Z"))


def add_raceway(bm, R, azimuth_deg, width, depth, z0, z1, mat=0):
    bm_box(bm, (R + depth / 2 - 0.05, 0, (z0 + z1) / 2), (depth + 0.1, width, z1 - z0),
           matrix=Matrix.Rotation(math.radians(azimuth_deg), 4, "Z"), mat=mat)


# =============================================================================
# Materials
# =============================================================================

def _set_input(node, names, value) -> bool:
    for n in (names,) if isinstance(names, str) else names:
        sock = node.inputs.get(n)
        if sock is not None:
            sock.default_value = value
            return True
    log.debug("Node %s has none of the inputs %s", node.bl_idname, names)
    return False


def _math(nt, op, a=None, b=None, c=None, clamp=False):
    n = nt.nodes.new("ShaderNodeMath")
    n.operation = op
    n.use_clamp = clamp
    for i, v in enumerate((a, b, c)):
        if v is None:
            continue
        if isinstance(v, (int, float)):
            n.inputs[i].default_value = v
        else:
            nt.links.new(v, n.inputs[i])
    return n.outputs[0]


def _map_range(nt, value, f_min, f_max, t_min, t_max, smooth=False):
    n = nt.nodes.new("ShaderNodeMapRange")
    n.clamp = True
    if smooth:
        n.interpolation_type = "SMOOTHSTEP"
    for name, v in (("Value", value), ("From Min", f_min), ("From Max", f_max),
                    ("To Min", t_min), ("To Max", t_max)):
        if isinstance(v, (int, float)):
            n.inputs[name].default_value = v
        else:
            nt.links.new(v, n.inputs[name])
    return n.outputs["Result"]


def _ramp(nt, fac, stops):
    n = nt.nodes.new("ShaderNodeValToRGB")
    els = n.color_ramp.elements
    els[0].position, els[0].color = stops[0][0], (*stops[0][1], 1.0)
    els[1].position, els[1].color = stops[-1][0], (*stops[-1][1], 1.0)
    for pos, col in stops[1:-1]:
        e = els.new(pos)
        e.color = (*col, 1.0)
    nt.links.new(fac, n.inputs["Fac"])
    return n.outputs["Color"]


def _principled(nt):
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
    nt.links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])
    out.location, bsdf.location = (400, 0), (100, 0)
    return bsdf


def _tidy(nt) -> None:
    """Spread nodes left-to-right in creation order so the graph is readable in the UI."""
    for i, n in enumerate(nt.nodes):
        if n.bl_idname not in ("ShaderNodeOutputMaterial", "ShaderNodeBsdfPrincipled"):
            n.location = (-250 * (1 + i // 4), -180 * (i % 4))


def build_steel_nodes(nt, M: MaterialParams, ring_height: float, soot: float) -> None:
    bsdf = _principled(nt)
    tc = nt.nodes.new("ShaderNodeTexCoord")
    obj = tc.outputs["Object"]
    # Brushed streaks: noise stretched along the vehicle axis
    mapping = nt.nodes.new("ShaderNodeMapping")
    s = M.steel_streak_scale
    mapping.inputs["Scale"].default_value = (s, s, s * 0.05)
    nt.links.new(obj, mapping.inputs["Vector"])
    noise = nt.nodes.new("ShaderNodeTexNoise")
    _set_input(noise, "Scale", 1.0)
    _set_input(noise, "Detail", 6.0)
    nt.links.new(mapping.outputs["Vector"], noise.inputs["Vector"])
    nfac = noise.outputs["Fac"]
    # Weld seams every ring_height metres
    sep = nt.nodes.new("ShaderNodeSeparateXYZ")
    nt.links.new(obj, sep.inputs["Vector"])
    z = sep.outputs["Z"]
    frac = _math(nt, "FRACT", _math(nt, "DIVIDE", z, ring_height))
    dist = _math(nt, "MULTIPLY", _math(nt, "SUBTRACT", 0.5, _math(nt, "ABSOLUTE", _math(nt, "SUBTRACT", frac, 0.5))), ring_height)
    seam = _map_range(nt, dist, 0.0, M.weld_seam_width, 1.0, 0.0, smooth=True)
    # Soot: strongest at the base, broken up by low-frequency noise
    soot_noise = nt.nodes.new("ShaderNodeTexNoise")
    _set_input(soot_noise, "Scale", 0.35)
    _set_input(soot_noise, "Detail", 4.0)
    nt.links.new(obj, soot_noise.inputs["Vector"])
    grad = _map_range(nt, z, 0.0, M.booster_soot_height, 1.0, 0.0, smooth=True)
    soot_mask = _math(nt, "MULTIPLY", _math(nt, "MULTIPLY", grad, soot_noise.outputs["Fac"]), soot * 2.0, clamp=True)
    # Colour and roughness
    f = _math(nt, "SUBTRACT", _math(nt, "MULTIPLY_ADD", nfac, 0.65, 0.35), soot_mask, clamp=True)
    sc = M.steel_color
    col = _ramp(nt, f, [(0.0, (0.06, 0.05, 0.045)), (0.35, tuple(c * 0.8 for c in sc)), (1.0, sc)])
    nt.links.new(col, bsdf.inputs["Base Color"])
    r0 = _map_range(nt, nfac, 0.3, 0.7, M.steel_roughness - 0.05, M.steel_roughness + 0.07)
    rough = _math(nt, "MAXIMUM", _math(nt, "MULTIPLY_ADD", soot_mask, 0.45, r0), _math(nt, "MULTIPLY", seam, 0.55))
    nt.links.new(rough, bsdf.inputs["Roughness"])
    # Bump: fine grain, then raised weld beads
    b1 = nt.nodes.new("ShaderNodeBump")
    _set_input(b1, "Strength", 0.015)
    nt.links.new(nfac, b1.inputs["Height"])
    b2 = nt.nodes.new("ShaderNodeBump")
    _set_input(b2, "Strength", M.weld_bump_strength)
    nt.links.new(seam, b2.inputs["Height"])
    nt.links.new(b1.outputs["Normal"], b2.inputs["Normal"])
    nt.links.new(b2.outputs["Normal"], bsdf.inputs["Normal"])
    # Circumferential anisotropy, as on rolled sheet
    tan = nt.nodes.new("ShaderNodeTangent")
    tan.direction_type = "RADIAL"
    tan.axis = "Z"
    if bsdf.inputs.get("Tangent") is not None:
        nt.links.new(tan.outputs["Tangent"], bsdf.inputs["Tangent"])
    _set_input(bsdf, "Metallic", 1.0)
    _set_input(bsdf, ("Anisotropic",), M.steel_anisotropy)
    _tidy(nt)


def build_tile_nodes(nt, M: MaterialParams) -> None:
    bsdf = _principled(nt)
    tc = nt.nodes.new("ShaderNodeTexCoord")
    edge = nt.nodes.new("ShaderNodeTexVoronoi")
    edge.feature = "DISTANCE_TO_EDGE"
    _set_input(edge, "Scale", 1.0 / max(M.tile_size, 0.01))
    nt.links.new(tc.outputs["Object"], edge.inputs["Vector"])
    cells = nt.nodes.new("ShaderNodeTexVoronoi")
    cells.feature = "F1"
    _set_input(cells, "Scale", 1.0 / max(M.tile_size, 0.01))
    nt.links.new(tc.outputs["Object"], cells.inputs["Vector"])
    bw = nt.nodes.new("ShaderNodeRGBToBW")
    nt.links.new(cells.outputs["Color"], bw.inputs["Color"])
    tile = _map_range(nt, edge.outputs["Distance"], 0.0, M.grout_width, 0.0, 1.0, smooth=True)
    f = _math(nt, "MULTIPLY", tile, _math(nt, "MULTIPLY_ADD", bw.outputs["Val"], 0.5, 0.5))
    tc_ = M.tile_color
    col = _ramp(nt, f, [(0.0, (0.035, 0.035, 0.035)), (0.5, tuple(c * 0.7 for c in tc_)),
                        (1.0, tuple(min(c * 1.6, 1.0) for c in tc_))])
    nt.links.new(col, bsdf.inputs["Base Color"])
    nt.links.new(_map_range(nt, tile, 0.0, 1.0, 0.9, M.tile_roughness), bsdf.inputs["Roughness"])
    bump = nt.nodes.new("ShaderNodeBump")
    _set_input(bump, "Strength", 0.35)
    nt.links.new(tile, bump.inputs["Height"])
    nt.links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])
    _set_input(bsdf, "Metallic", 0.0)
    _set_input(bsdf, ("Specular IOR Level", "Specular"), 0.18)
    _tidy(nt)


def build_bell_nodes(nt, M: MaterialParams) -> None:
    """Heat-tinted nozzle: grey at the powerhead, bronze then blue-violet toward the exit."""
    bsdf = _principled(nt)
    tc = nt.nodes.new("ShaderNodeTexCoord")
    sep = nt.nodes.new("ShaderNodeSeparateXYZ")
    nt.links.new(tc.outputs["Generated"], sep.inputs["Vector"])
    col = _ramp(nt, sep.outputs["Z"], [(0.0, (0.12, 0.12, 0.17)), (0.3, (0.32, 0.22, 0.32)),
                                       (0.6, (0.62, 0.44, 0.28)), (0.85, (0.55, 0.55, 0.57)),
                                       (1.0, (0.5, 0.5, 0.52))])
    nt.links.new(col, bsdf.inputs["Base Color"])
    _set_input(bsdf, "Metallic", 1.0)
    _set_input(bsdf, "Roughness", M.bell_roughness)
    _tidy(nt)


def build_flat_nodes(nt, color, metallic, roughness) -> None:
    bsdf = _principled(nt)
    _set_input(bsdf, "Base Color", (*color, 1.0))
    _set_input(bsdf, "Metallic", metallic)
    _set_input(bsdf, "Roughness", roughness)


def ensure_material(ctx: BuildContext, name: str, key: str, build: Callable, viewport_color) -> bpy.types.Material:
    mat = bpy.data.materials.get(name)
    if ctx.is_fresh(mat, key):
        ctx.bump("materials_reused")
        return mat
    if mat is None:
        mat = bpy.data.materials.new(name)
        mat[TAG] = True
    if mat.node_tree is None:
        mat.use_nodes = True  # pre-5.0 only; 5.x always has a node tree
    nt = mat.node_tree
    nt.nodes.clear()
    build(nt)
    mat.diffuse_color = (*viewport_color, 1.0)
    mat[HASH_KEY] = key
    ctx.bump("materials_built")
    return mat


def build_materials(ctx: BuildContext, P: StarshipParams) -> Dict[str, bpy.types.Material]:
    M, rh = P.materials, P.hull.ring_height
    specs = {
        "steel_ship": (lambda nt: build_steel_nodes(nt, M, rh, 0.0), M.steel_color,
                       ("steel", M.steel_color, M.steel_roughness, M.steel_anisotropy, M.steel_streak_scale,
                        M.weld_seam_width, M.weld_bump_strength, M.booster_soot_height, rh, 0.0)),
        "steel_booster": (lambda nt: build_steel_nodes(nt, M, rh, M.booster_soot), tuple(c * 0.85 for c in M.steel_color),
                          ("steel", M.steel_color, M.steel_roughness, M.steel_anisotropy, M.steel_streak_scale,
                           M.weld_seam_width, M.weld_bump_strength, M.booster_soot_height, rh, M.booster_soot)),
        "heat_tile": (lambda nt: build_tile_nodes(nt, M), (0.03, 0.03, 0.03),
                      ("tile", M.tile_color, M.tile_roughness, M.tile_size, M.grout_width)),
        "engine_bell": (lambda nt: build_bell_nodes(nt, M), (0.45, 0.38, 0.33), ("bell", M.bell_roughness)),
        "dark_metal": (lambda nt: build_flat_nodes(nt, (0.06, 0.06, 0.065), 0.85, 0.45), (0.06, 0.06, 0.065), ("dark",)),
        "aperture": (lambda nt: build_flat_nodes(nt, (0.004, 0.004, 0.004), 0.0, 1.0), (0.0, 0.0, 0.0), ("aperture",)),
    }
    return {role: ensure_material(ctx, f"{PREFIX}MAT_{role}", param_hash(*key), build, vp)
            for role, (build, vp, key) in specs.items()}


# =============================================================================
# Geometry
# =============================================================================

@dataclass
class Hinge:
    obj: bpy.types.Object
    axis: int            # 0/1/2 = local X/Y/Z rotation channel
    role: str            # forward_flap | aft_flap | grid_fin | payload_door
    sign: float = 1.0    # direction that "stows"/"opens" the panel
    side: float = 1.0    # +1 / -1 for differential motion
    index: int = 0


@dataclass
class Rig:
    stack: Optional[bpy.types.Object] = None
    ship_root: Optional[bpy.types.Object] = None
    booster_root: Optional[bpy.types.Object] = None
    hinges: List[Hinge] = field(default_factory=list)


def make_hinge(ctx, name: str, coll, parent, location, mount_euler, axis: int, rest: float = 0.0):
    """Two-level pivot: a static Mount carries the orientation, the Hinge rotates on one axis."""
    mount, _ = ensure_empty(ctx, f"{name}_Mount", coll, parent, "ARROWS", 0.8)
    mount.location = location
    mount.rotation_mode = "XYZ"
    mount.rotation_euler = mount_euler
    hinge, created = ensure_empty(ctx, f"{name}_Hinge", coll, mount, "SINGLE_ARROW", 1.2)
    hinge.location = (0, 0, 0)
    hinge.rotation_mode = "XYZ"
    hinge["hinge_axis"] = "XYZ"[axis]
    if created:
        rot = [0.0, 0.0, 0.0]
        rot[axis] = rest
        hinge.rotation_euler = rot
    return mount, hinge


def build_engine_cluster(ctx, P, mats, coll, parent, prefix: str, layout, mount_z: float) -> None:
    """layout: list of (kind, count, radius, phase). Engines share one mesh per kind (linked data)."""
    E = P.engines
    shared = {}
    for kind, length, exit_r in (("SL", E.sl_length, E.sl_exit_radius), ("VAC", E.vac_length, E.vac_exit_radius)):
        if any(k == kind for k, *_ in layout):
            shared[kind] = ensure_mesh(ctx, f"{PREFIX}MESH_Raptor_{kind}", param_hash(E, kind),
                                       lambda bm, L=length, r=exit_r: build_engine_bm(bm, E, L, r),
                                       [mats["engine_bell"], mats["dark_metal"]])
    n = 0
    for kind, count, radius, phase in layout:
        for i in range(count):
            a = phase + 2 * math.pi * i / count
            obj, _ = ensure_object(ctx, f"{prefix}_Engine_{kind}_{n:02d}", shared[kind], coll, parent)
            obj.location = (radius * math.cos(a), radius * math.sin(a), mount_z)
            obj.rotation_euler = (0, 0, a + math.pi / 2)  # pumps tangential, never through the hull
            ensure_modifier(obj, "Wall", "SOLIDIFY", thickness=E.wall_thickness, offset=0.0)
            n += 1


def build_ship(ctx, P: StarshipParams, mats, coll, stack, rig: Rig) -> None:
    R, S, F, D = P.hull.radius, P.ship, P.flaps, P.payload_door
    seg = P.hull.segments
    ship_root, _ = ensure_empty(ctx, f"{PREFIX}Ship_Root", coll, stack, "CUBE", 2.0)
    ship_root.location = (0, 0, P.booster.length if P.include_booster else 0.0)
    rig.ship_root = ship_root
    profile = ship_profile(P)
    nose_base = S.length - S.nose_length

    def hull(bm):
        bm_lathe(bm, profile, seg, mat_fn=windward_tile_fn(S.tile_arc_deg))
        bm_lathe(bm, [(0.0, S.engine_mount_z + 0.05), (R - 0.02, S.engine_mount_z + 0.05)], seg, mat=2, smooth=False)
        if S.raceway:
            add_raceway(bm, R, S.raceway_azimuth_deg, S.raceway_width, S.raceway_depth, 0.8, nose_base)

    mesh = ensure_mesh(ctx, f"{PREFIX}MESH_Ship_Hull", param_hash(P.hull, geometry_fields(S, "rsl_count", "rsl_ring_radius", "rvac_count", "rvac_ring_radius")), hull,
                       [mats["steel_ship"], mats["heat_tile"], mats["dark_metal"]])
    ensure_object(ctx, f"{PREFIX}Ship_Hull", mesh, coll, ship_root)

    v_phase = math.pi / max(S.rvac_count, 1)
    build_engine_cluster(ctx, P, mats, coll, ship_root, f"{PREFIX}Ship",
                         [("SL", S.rsl_count, S.rsl_ring_radius, 0.0),
                          ("VAC", S.rvac_count, S.rvac_ring_radius, v_phase)], S.engine_mount_z)

    if F.enabled:
        fr = F.fairing_radius
        flap_specs = (
            ("Fwd", "forward_flap", F.forward_hinge_z, F.forward_root_chord, F.forward_tip_chord,
             F.forward_span, F.forward_tip_shift, F.forward_windward_offset_deg),
            ("Aft", "aft_flap", F.aft_hinge_z, F.aft_root_chord, F.aft_tip_chord,
             F.aft_span, F.aft_tip_shift, F.aft_windward_offset_deg),
        )
        for label, role, z0, root, tip, span, shift, offset in flap_specs:
            r0 = profile_radius_at(profile, z0)
            r1 = profile_radius_at(profile, z0 + root)
            beta = math.atan2(r0 - r1, root)          # hinge follows the nose taper
            for side_label, side, az_deg in (("R", 1.0, -offset), ("L", -1.0, 180.0 + offset)):
                az = math.radians(az_deg)
                stow_sign = 1.0 if math.cos(az) >= 0 else -1.0     # fold toward leeward +Y
                windward_sign = -int(stow_sign)                    # which local-Y face meets the flow
                name = f"{PREFIX}Ship_{label}Flap_{side_label}"
                rr = r0 + fr * 0.5
                mount, hinge = make_hinge(ctx, name, coll, ship_root,
                                          (rr * math.cos(az), rr * math.sin(az), z0),
                                          (0.0, -beta, az), axis=2,
                                          rest=math.radians(F.stowed_angle_deg) * stow_sign)
                fmesh = ensure_mesh(ctx, f"{PREFIX}MESH_{label}Flap_{side_label}",
                                    param_hash(label, root, tip, span, shift, F.thickness, fr, windward_sign),
                                    lambda bm, a=(root, tip, span, shift, windward_sign):
                                    build_flap_bm(bm, a[0], a[1], a[2], a[3], F.thickness, fr * 0.6, a[4]),
                                    [mats["steel_ship"], mats["heat_tile"]], sharp_angle_deg=None)
                flap, _ = ensure_object(ctx, f"{name}_Panel", fmesh, coll, hinge)
                ensure_modifier(flap, "Bevel", "BEVEL", width=0.04, segments=2, limit_method="ANGLE")
                fair = ensure_mesh(ctx, f"{PREFIX}MESH_{label}Fairing", param_hash("fairing", root, fr),
                                   lambda bm, L=root: bm_cylinder(bm, fr, L, 24, Matrix.Translation((0, 0, L / 2))),
                                   [mats["steel_ship"]])
                ensure_object(ctx, f"{name}_Fairing", fair, coll, mount)
                rig.hinges.append(Hinge(hinge, 2, role, stow_sign, side))

    if D.enabled:
        az = math.radians(D.azimuth_deg)
        half = (D.width / 2) / R
        origin = Vector((R, 0, D.top_z))
        name = f"{PREFIX}Ship_PayloadDoor"
        mount, hinge = make_hinge(ctx, name, coll, ship_root,
                                  (R * math.cos(az), R * math.sin(az), D.top_z), (0, 0, az), axis=1)
        dmesh = ensure_mesh(ctx, f"{PREFIX}MESH_PayloadDoor", param_hash(D.width, D.height, D.thickness, R),
                            lambda bm: bm_curved_panel(bm, R + D.thickness / 2 + 0.03, half, D.top_z - D.height,
                                                       D.top_z, 16, origin), [mats["steel_ship"]])
        door, _ = ensure_object(ctx, f"{name}_Panel", dmesh, coll, hinge)
        ensure_modifier(door, "Thickness", "SOLIDIFY", thickness=D.thickness, offset=0.0)
        amesh = ensure_mesh(ctx, f"{PREFIX}MESH_PayloadAperture", param_hash(D.width, D.height, R),
                            lambda bm: bm_curved_panel(bm, R + 0.01, half * 0.96, D.top_z - D.height * 0.96,
                                                       D.top_z - D.height * 0.04, 16, origin), [mats["aperture"]])
        ensure_object(ctx, f"{name}_Aperture", amesh, coll, mount)
        rig.hinges.append(Hinge(hinge, 1, "payload_door", -1.0))


def build_booster(ctx, P: StarshipParams, mats, coll, stack, rig: Rig) -> None:
    R, B, G, seg = P.hull.radius, P.booster, P.grid_fins, P.hull.segments
    root, _ = ensure_empty(ctx, f"{PREFIX}Booster_Root", coll, stack, "CUBE", 2.0)
    root.location = (0, 0, 0)
    rig.booster_root = root
    tank_top = B.length - B.hot_stage_height
    mount_z = B.engine_exit_z + P.engines.sl_length

    def hull(bm):
        dome_h = B.hot_stage_height * 0.5
        dome = [(R * math.cos(math.radians(a)), tank_top + dome_h * math.sin(math.radians(a))) for a in range(0, 91, 10)]
        bm_lathe(bm, [(R, 0.0)] + dome, seg, mat=0)
        bm_lathe(bm, [(0.0, mount_z + 0.05), (R - 0.02, mount_z + 0.05)], seg, mat=1, smooth=False)
        if B.raceway:
            add_raceway(bm, R, B.raceway_azimuth_deg, B.raceway_width, B.raceway_depth, 1.0, tank_top - 0.5)

    mesh = ensure_mesh(ctx, f"{PREFIX}MESH_Booster_Hull", param_hash(P.hull, geometry_fields(B, "engine_rings", "hot_stage_struts", "hot_stage_strut_width", "hot_stage_band"), P.engines.sl_length), hull,
                       [mats["steel_booster"], mats["dark_metal"]])
    ensure_object(ctx, f"{PREFIX}Booster_Hull", mesh, coll, root)

    ring = ensure_mesh(ctx, f"{PREFIX}MESH_Booster_HotStage", param_hash(P.hull.radius, P.hull.segments, B.hot_stage_height, B.hot_stage_struts, B.hot_stage_strut_width, B.hot_stage_band),
                       lambda bm: build_hot_stage_bm(bm, R, B, seg), [mats["steel_booster"]])
    hs, _ = ensure_object(ctx, f"{PREFIX}Booster_HotStage", ring, coll, root)
    hs.location = (0, 0, tank_top)

    layout = [("SL", count, radius, (math.pi / count) * (k % 2)) for k, (count, radius) in enumerate(B.engine_rings)]
    build_engine_cluster(ctx, P, mats, coll, root, f"{PREFIX}Booster", layout, mount_z)

    if G.enabled and G.azimuths_deg:
        fin_mesh = ensure_mesh(ctx, f"{PREFIX}MESH_GridFin", param_hash(geometry_fields(G, "enabled", "azimuths_deg", "z_below_tank_top")), lambda bm: build_grid_fin_bm(bm, G),
                               [mats["steel_booster"], mats["dark_metal"]], sharp_angle_deg=None)
        housing = ensure_mesh(ctx, f"{PREFIX}MESH_GridFinHousing", param_hash(G.width, G.chord),
                              lambda bm: bm_box(bm, (0.15, 0, 0), (0.7, G.width * 0.5, G.chord * 3.0), mat=0),
                              [mats["steel_booster"]], sharp_angle_deg=None)
        z = tank_top - G.z_below_tank_top
        for i, az_deg in enumerate(G.azimuths_deg):
            az = math.radians(az_deg)
            name = f"{PREFIX}Booster_GridFin_{i:02d}"
            mount, hinge = make_hinge(ctx, name, coll, root, (R * math.cos(az), R * math.sin(az), z), (0, 0, az), axis=0)
            ensure_object(ctx, f"{name}_Fin", fin_mesh, coll, hinge)
            ensure_object(ctx, f"{name}_Housing", housing, coll, mount)
            rig.hinges.append(Hinge(hinge, 0, "grid_fin", 1.0, 1.0 if i % 2 == 0 else -1.0, i))


def build_geometry(ctx: BuildContext, P: StarshipParams, mats, report: "RunReport") -> Rig:
    root_coll = ensure_collection("Starship")
    rig = Rig()
    stack, _ = ensure_empty(ctx, f"{PREFIX}Stack", root_coll, None, "CUBE", 5.0)
    rig.stack = stack
    with report.stage("geometry.ship", ctx):
        build_ship(ctx, P, mats, ensure_collection("Starship.Ship", root_coll), stack, rig)
    if P.include_booster:
        with report.stage("geometry.booster", ctx):
            build_booster(ctx, P, mats, ensure_collection("Starship.Booster", root_coll), stack, rig)
    return rig


# =============================================================================
# Camera, environment, render
# =============================================================================

def setup_camera(ctx: BuildContext, P: StarshipParams) -> bpy.types.Object:
    """Orbit rig: Orbit empty (animated Z) -> Camera tracking a Target empty. Returns the orbit empty."""
    C = P.camera
    coll = ensure_collection("Starship.Rig", ensure_collection("Starship"))
    target, _ = ensure_empty(ctx, f"{PREFIX}Cam_Target", coll, None, "SPHERE", 2.0)
    target.location = (0, 0, C.target_z)
    orbit, _ = ensure_empty(ctx, f"{PREFIX}Cam_Orbit", coll, None, "CIRCLE", 10.0)
    cam_data = bpy.data.cameras.get(f"{PREFIX}Camera") or bpy.data.cameras.new(f"{PREFIX}Camera")
    cam_data[TAG] = True
    cam_data.lens = C.lens_mm
    cam_data.clip_end = C.clip_end
    cam_data.dof.use_dof = C.use_dof
    cam_data.dof.focus_object = target
    cam_data.dof.aperture_fstop = C.f_stop
    cam, _ = ensure_object(ctx, f"{PREFIX}Camera", cam_data, coll, orbit)
    a = math.radians(C.start_azimuth_deg)
    cam.location = (C.distance * math.cos(a), C.distance * math.sin(a), C.height)
    con = cam.constraints.get("Track") or cam.constraints.new("TRACK_TO")
    con.name = "Track"
    con.target = target
    con.track_axis = "TRACK_NEGATIVE_Z"
    con.up_axis = "UP_Y"
    bpy.context.scene.camera = cam
    return orbit


def setup_environment(ctx: BuildContext, P: StarshipParams) -> None:
    E = P.environment
    scene = bpy.context.scene
    coll = ensure_collection("Starship.Rig", ensure_collection("Starship"))
    sun_data = bpy.data.lights.get(f"{PREFIX}Sun") or bpy.data.lights.new(f"{PREFIX}Sun", "SUN")
    sun_data[TAG] = True
    sun_data.energy = E.sun_strength
    sun_data.angle = math.radians(E.sun_angle_deg)
    sun, _ = ensure_object(ctx, f"{PREFIX}Sun", sun_data, coll, None)
    el, az = math.radians(E.sun_elevation_deg), math.radians(E.sun_azimuth_deg)
    to_sun = Vector((math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)))
    sun.rotation_mode = "XYZ"
    sun.rotation_euler = to_sun.to_track_quat("Z", "Y").to_euler()

    world = bpy.data.worlds.get(f"{PREFIX}World")
    key = param_hash(E)
    if world is None:
        world = bpy.data.worlds.new(f"{PREFIX}World")
        world[TAG] = True
    scene.world = world
    if ctx.is_fresh(world, key):
        ctx.bump("materials_reused")
        return
    if world.node_tree is None:
        world.use_nodes = True
    nt = world.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputWorld")
    bg = nt.nodes.new("ShaderNodeBackground")
    nt.links.new(bg.outputs["Background"], out.inputs["Surface"])
    bg.inputs["Strength"].default_value = E.sky_strength
    sky = nt.nodes.new("ShaderNodeTexSky")
    available = sky.bl_rna.properties["sky_type"].enum_items.keys()
    chosen = next((t for t in ("MULTIPLE_SCATTERING", "NISHITA", "SINGLE_SCATTERING") if t in available), None)
    if chosen:
        sky.sky_type = chosen
        for attr, val in (("sun_disc", False), ("sun_elevation", el), ("sun_rotation", az)):
            if hasattr(sky, attr):
                setattr(sky, attr, val)
        nt.links.new(sky.outputs["Color"], bg.inputs["Color"])
        log.info("World: physical sky (%s)", chosen)
    else:
        nt.nodes.remove(sky)
        bg.inputs["Color"].default_value = (*E.fallback_color, 1.0)
        log.info("World: physical sky unavailable, using flat colour")
    world[HASH_KEY] = key
    ctx.bump("materials_built")


def configure_render(P: StarshipParams) -> None:
    R, A, scene = P.render, P.animation, bpy.context.scene
    candidates = ["CYCLES"] if R.engine.upper() == "CYCLES" else ["BLENDER_EEVEE", "BLENDER_EEVEE_NEXT"]
    for eng in candidates:
        try:
            scene.render.engine = eng
            break
        except TypeError:
            continue
    if scene.render.engine == "CYCLES":
        scene.cycles.samples = R.samples
        scene.cycles.use_denoising = R.denoise
    elif hasattr(scene, "eevee"):
        for attr in ("taa_render_samples",):
            if hasattr(scene.eevee, attr):
                setattr(scene.eevee, attr, R.samples)
    scene.render.resolution_x = R.resolution_x
    scene.render.resolution_y = R.resolution_y
    scene.render.resolution_percentage = R.resolution_percentage
    scene.render.fps = A.fps
    scene.frame_start, scene.frame_end = A.frame_start, A.frame_end
    for vt in ("AgX", "Filmic"):
        try:
            scene.view_settings.view_transform = vt
            break
        except TypeError:
            continue


# =============================================================================
# Animation
# =============================================================================

@contextmanager
def _eased_keyframes():
    """New keys get Bezier/auto-clamped handles (ease in/out) on every Blender version."""
    edit = bpy.context.preferences.edit
    old = (edit.keyframe_new_interpolation_type, edit.keyframe_new_handle_type)
    edit.keyframe_new_interpolation_type, edit.keyframe_new_handle_type = "BEZIER", "AUTO_CLAMPED"
    try:
        yield
    finally:
        edit.keyframe_new_interpolation_type, edit.keyframe_new_handle_type = old


def _key_rot(obj, axis: int, frame: int, value: float) -> None:
    obj.rotation_euler[axis] = value
    obj.keyframe_insert("rotation_euler", index=axis, frame=frame)


def _limit_hinge(h: Hinge, values: Sequence[float]) -> None:
    """Hard stops matching the keyed range, so manual posing can't over-rotate a hinge."""
    con = h.obj.constraints.get("HingeStops") or h.obj.constraints.new("LIMIT_ROTATION")
    con.name = "HingeStops"
    con.owner_space = "LOCAL"
    ax = "xyz"[h.axis]
    pad = math.radians(2.0)
    setattr(con, f"use_limit_{ax}", True)
    setattr(con, f"min_{ax}", min(values) - pad)
    setattr(con, f"max_{ax}", max(values) + pad)


def animate(ctx: BuildContext, P: StarshipParams, rig: Rig, orbit: Optional[bpy.types.Object]) -> None:
    A, F = P.animation, P.flaps
    scene = bpy.context.scene
    animated = [h.obj for h in rig.hinges] + [o for o in (rig.ship_root, orbit) if o is not None]
    key = param_hash(A, F, P.grid_fins, P.payload_door, P.include_booster, P.booster.length,
                     sorted(o.name for o in animated))
    if not A.enabled:
        for o in animated:
            o.animation_data_clear()
        scene[HASH_KEY + "_anim"] = ""
        return
    if not ctx.force and scene.get(HASH_KEY + "_anim") == key and all(o.animation_data for o in animated):
        ctx.bump("animations_reused")
        return
    for o in animated:
        o.animation_data_clear()

    s, e = A.frame_start, A.frame_end
    with _eased_keyframes():
        for h in rig.hinges:
            keyed: List[float] = []

            def k(frame, value, h=h, keyed=keyed):
                if s <= frame <= e:
                    _key_rot(h.obj, h.axis, frame, value)
                    keyed.append(value)

            if h.role in ("forward_flap", "aft_flap"):
                stow = math.radians(F.stowed_angle_deg) * h.sign
                dep = math.radians(F.deployed_angle_deg) * h.sign
                k(s, stow)
                k(A.flap_deploy_start, stow)
                k(A.flap_deploy_end, dep)
                phase_off = 1.0 if h.role == "forward_flap" else -1.0   # pitch: fwd vs aft opposite
                n, f = 0, A.flap_deploy_end + A.flap_flutter_interval
                while f <= e:
                    delta = math.radians(A.flap_flutter_deg) * (1 if n % 2 == 0 else -1) * h.side * phase_off
                    k(f, dep + h.sign * delta)
                    n, f = n + 1, f + A.flap_flutter_interval
            elif h.role == "grid_fin":
                step = max(A.gridfin_period // 4, 1)
                k(s, 0.0)
                f = A.flap_deploy_start
                while f <= e:
                    phase = 2 * math.pi * (f - A.flap_deploy_start) / A.gridfin_period
                    k(f, math.radians(A.gridfin_sweep_deg) * math.sin(phase) * h.side)
                    f += step
            elif h.role == "payload_door":
                k(s, 0.0)
                k(A.door_open_start, 0.0)
                k(A.door_open_end, h.sign * math.radians(A.door_open_deg))
            if keyed:
                _limit_hinge(h, keyed)

        if rig.ship_root is not None and P.include_booster and A.separation_distance > 0:
            z0 = rig.ship_root.location.z
            for frame, z in ((s, z0), (A.separation_start, z0), (A.separation_end, z0 + A.separation_distance)):
                if s <= frame <= e:
                    rig.ship_root.location.z = z
                    rig.ship_root.keyframe_insert("location", index=2, frame=frame)
            rig.ship_root.location.z = z0
        if orbit is not None and A.camera_orbit_deg:
            _key_rot(orbit, 2, s, 0.0)
            _key_rot(orbit, 2, e, math.radians(A.camera_orbit_deg))

    scene[HASH_KEY + "_anim"] = key
    scene.frame_set(s)
    ctx.bump("animations_built")


# =============================================================================
# Reporting: timings and AI usage cost
# =============================================================================

class RunReport:
    def __init__(self):
        self.t0 = time.perf_counter()
        self.stages: List[dict] = []

    @contextmanager
    def stage(self, name: str, ctx: BuildContext):
        before = dict(ctx.counters)
        t = time.perf_counter()
        try:
            yield
        finally:
            dt = time.perf_counter() - t
            delta = {k: v - before[k] for k, v in ctx.counters.items() if v != before[k]}
            self.stages.append({"stage": name, "seconds": round(dt, 4), **delta})
            log.info("%-22s %8.3f s  %s", name, dt,
                     " ".join(f"{k}={v}" for k, v in delta.items()) or "(no changes)")

    def total(self) -> float:
        return time.perf_counter() - self.t0


def report_ai_usage() -> dict:
    """
    This generator makes no model or API calls, so its own AI cost is exactly zero.
    The cost of the AI session that *wrote or tuned* the script is not visible from
    inside Blender. To include it, point STARSHIP_GEN_AI_LEDGER at a JSON file:
        [{"source": "...", "input_tokens": 0, "output_tokens": 0, "usd": 0.0}, ...]
    Figures are summed and logged verbatim; nothing is estimated or invented.
    """
    result = {"runtime_ai_calls": 0, "runtime_ai_cost_usd": 0.0, "ledger": None}
    path = os.environ.get("STARSHIP_GEN_AI_LEDGER")
    if not path:
        log.info("AI usage: script made 0 AI calls (US$0.00). External session cost: not available "
                 "(set STARSHIP_GEN_AI_LEDGER to log it).")
        return result
    try:
        with open(path, encoding="utf-8") as fh:
            entries = json.load(fh)
        if isinstance(entries, dict):
            entries = entries.get("entries", [])
        total = {
            "entries": len(entries),
            "input_tokens": sum(int(e.get("input_tokens", 0)) for e in entries),
            "output_tokens": sum(int(e.get("output_tokens", 0)) for e in entries),
            "usd": round(sum(float(e.get("usd", 0.0)) for e in entries), 6),
        }
        result["ledger"] = {"path": path, **total}
        log.info("AI usage ledger %s: %d entries, %d in / %d out tokens, US$%.4f",
                 path, total["entries"], total["input_tokens"], total["output_tokens"], total["usd"])
    except (OSError, ValueError, TypeError, AttributeError) as exc:
        result["ledger"] = {"path": path, "error": str(exc)}
        log.warning("AI usage ledger %s unreadable: %s", path, exc)
    return result


# =============================================================================
# Entry points
# =============================================================================

def generate(params: Optional[StarshipParams] = None, force: bool = False) -> dict:
    """Build or update the scene. Returns a machine-readable report."""
    P = params or StarshipParams()
    validate(P)
    ctx, report = BuildContext(force=force), RunReport()
    log.info("Starship generator %s on Blender %s (force=%s)", GENERATOR_VERSION, bpy.app.version_string, force)
    with report.stage("materials", ctx):
        mats = build_materials(ctx, P)
    with report.stage("geometry", ctx):
        rig = build_geometry(ctx, P, mats, report)
    with report.stage("camera_environment", ctx):
        orbit = setup_camera(ctx, P)
        setup_environment(ctx, P)
        configure_render(P)
    with report.stage("animation", ctx):
        animate(ctx, P, rig, orbit)
    with report.stage("cleanup", ctx):
        garbage_collect(ctx)
    total = report.total()
    log.info("Total %.3f s | built %d meshes / reused %d | built %d materials / reused %d | %d hinges",
             total, ctx.counters["meshes_built"], ctx.counters["meshes_reused"],
             ctx.counters["materials_built"], ctx.counters["materials_reused"], len(rig.hinges))
    return {
        "generator_version": GENERATOR_VERSION,
        "blender_version": bpy.app.version_string,
        "params_hash": param_hash(P),
        "total_seconds": round(total, 4),
        "stages": report.stages,
        "counters": ctx.counters,
        "hinges": [{"object": h.obj.name, "role": h.role, "axis": "XYZ"[h.axis]} for h in rig.hinges],
        "ai_usage": report_ai_usage(),
    }


def _parse_args(argv: List[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="starship_generator", description="Procedural Starship stack for Blender.")
    p.add_argument("--params", help="JSON file of parameter overrides (partial is fine)")
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                   help="override one parameter, e.g. --set ship.length=50 --set grid_fins.azimuths_deg=[0,120,240]")
    p.add_argument("--force", action="store_true", help="rebuild everything, ignoring cached hashes")
    p.add_argument("--dump-params", action="store_true", help="print the resolved parameters as JSON and exit")
    p.add_argument("--remove-default-cube", action="store_true",
                   help="delete the factory-startup 'Cube' if present (never touches generated objects)")
    p.add_argument("--save", help="save the .blend to this path")
    p.add_argument("--render", help="render a still (PNG path) or, with --animation, a frame-path prefix")
    p.add_argument("--frame", type=int, help="frame to render for a still")
    p.add_argument("--animation", action="store_true", help="render the full frame range")
    p.add_argument("--log-json", help="write the run report (timings, counters, AI usage) to this JSON file")
    p.add_argument("--log-level", default="INFO")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> dict:
    argv = sys.argv if argv is None else argv
    args = _parse_args(argv[argv.index("--") + 1:] if "--" in argv else [])
    if not log.handlers:
        h = logging.StreamHandler(sys.stdout)
        h.setFormatter(logging.Formatter("[starship] %(levelname)s %(message)s"))
        log.addHandler(h)
    log.setLevel(args.log_level.upper())
    log.propagate = False

    P = StarshipParams()
    if args.params:
        with open(args.params, encoding="utf-8") as fh:
            apply_overrides(P, json.load(fh))
    for item in args.set:
        k, _, v = item.partition("=")
        set_param(P, k.strip(), v.strip())
    if args.dump_params:
        print(json.dumps(asdict(P), indent=2))
        return {}

    if args.remove_default_cube:
        cube = bpy.data.objects.get("Cube")
        if cube is not None and not cube.get(TAG):
            bpy.data.objects.remove(cube, do_unlink=True)

    result = generate(P, force=args.force)
    scene = bpy.context.scene
    if args.save:
        bpy.ops.wm.save_as_mainfile(filepath=os.path.abspath(args.save))
        log.info("Saved %s", args.save)
    if args.render:
        t = time.perf_counter()
        scene.render.filepath = os.path.abspath(args.render)
        if args.animation:
            bpy.ops.render.render(animation=True)
        else:
            scene.frame_set(args.frame if args.frame is not None else scene.frame_start)
            scene.render.image_settings.file_format = "PNG"
            bpy.ops.render.render(write_still=True)
        result["render_seconds"] = round(time.perf_counter() - t, 3)
        log.info("Rendered to %s in %.2f s", args.render, result["render_seconds"])
    if args.log_json:
        with open(args.log_json, "w", encoding="utf-8") as fh:
            json.dump(result, fh, indent=2)
    return result


if __name__ == "__main__":
    main()
