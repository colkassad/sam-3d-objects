# Ouster route recordings to positioned meshes

`scripts/ouster_prompt_to_scene.py` turns a recorded, single-sensor Ouster PCAP
or OSF into one prompted SAM 3D mesh per associated static object. KISS SLAM
runs on every scan, while SAM 3 runs only on motion-selected keyframes and SAM
3D runs only once for each accepted track.

The first version expects `ChanField.RGB` to be pixel-registered with `RANGE`.
It targets bounded, static objects such as vehicles, signs, poles, equipment,
and trees. Tracks are classified as confirmed static, dynamic, or unconfirmed;
only confirmed-static tracks are meshed. Two observations can establish
motion, while a singleton remains unconfirmed instead of being assumed static.

## Environment setup

Install the pinned Ouster SDK into the SAM 3D environment, then refresh both
editable installs so their new commands are available:

```bash
micromamba run -n sam3d-objects \
  pip install -r requirements.ouster.txt
micromamba run -n sam3d-objects \
  pip install -e . --no-deps
micromamba run -n sam3-masking \
  pip install -e . --no-deps
```

The route orchestrator discovers `sam3-mask-route` in the sibling
`sam3-masking` environment. It can also be selected with
`--sam3-executable` or `SAM3_MASK_ROUTE_EXECUTABLE`.

## End-to-end use

```bash
micromamba run -n sam3d-objects \
  python scripts/ouster_prompt_to_scene.py run /data/route.osf \
  --output-dir outputs/route \
  --prompt "parked car" \
  --prompt "traffic sign" \
  --point-cloud route.ply
```

For a PCAP whose metadata cannot be discovered automatically, add
`--meta /data/route.json`.

OSF recordings can be processed from a bounded frame window without reading
the preceding scans:

```bash
python scripts/ouster_prompt_to_scene.py run /data/route.osf \
  --output-dir outputs/window \
  --start-frame 1001 \
  --stop-frame 1500 \
  --prompt "vehicle"
```

Frame bounds are 1-based and inclusive, matching `scan_index` and keyframe
names. Omitting the start uses frame 1; omitting the stop uses the final OSF
frame. KISS SLAM starts fresh at the effective start frame, so the window has
its own local world origin. If `--max-scans` is also present, it limits scans
inside the window. Frame bounds are not accepted for PCAP inputs.

The default keyframe trigger is 5 metres or 5 degrees since the previous
keyframe. The route-oriented SAM 3D defaults are 10,000 faces, 15 sampling
steps in each diffusion stage, flat shading, and the low-VRAM memory profile.
Face decimation reduces output complexity; the reduced sampling steps provide
the inference-time saving.

Mesh reconstruction is limited by default to tracks with at least one cleaned
observation at or below 30 metres median LiDAR range. Far observations still
contribute to association and motion classification, but they cannot supply
SAM 3D imagery or metric fitting points. Change the limit with
`--max-mesh-range-m`; use `--no-max-mesh-range` to disable it.

Position fitting is grounded by default. SAM 3D's PyTorch3D row-vector pose is
converted to the GLB/world column-vector convention, GLB local Y is checked
against SLAM world-up, and elongated meshes are aligned to the robust
horizontal principal axis of their aggregated LiDAR points. The default
`--fit-mode raycast` reconstructs the original per-pixel LiDAR rays from every
reliable, range-qualified observation and compares their measured ranges with
the first mesh intersections. It can adjust yaw, translation, and bounded
per-axis scale, but a candidate is accepted only when it improves the original
SAM 3D pointmap pose. This avoids shrinking a complete mesh onto a dense,
one-sided visible surface. Yaw is polished separately using observations that
expose at least 45% of the fused longitudinal support; nearly end-on views can
still constrain range and translation but cannot dominate heading. Informative
views estimate heading from robust local surface tangents, combining
perpendicular footprint faces modulo 90 degrees instead of using a
viewpoint-biased global PCA axis. They also receive stricter depth and
mask-border regression checks. Use
`--fit-mode none` to retain only the upright,
pointmap-derived pose. Ray count, view count, scale-change, and optimization
limits are controlled by `--fit-max-rays-per-view`, `--fit-max-views`,
`--fit-max-axis-scale-change`, and `--fit-max-evaluations`.

## Staged and resumable use

```bash
python scripts/ouster_prompt_to_scene.py extract /data/route.osf \
  --output-dir outputs/route

python scripts/ouster_prompt_to_scene.py segment outputs/route \
  --prompt "parked car" \
  --prompt "traffic sign"

python scripts/ouster_prompt_to_scene.py reconstruct outputs/route
```

After changing tracking code or motion thresholds, rebuild range hypotheses
and tracks from the existing SAM 3 masks without loading SAM 3 again:

```bash
python scripts/ouster_prompt_to_scene.py track outputs/route \
  --max-mesh-range-m 30 \
  --overwrite
python scripts/ouster_prompt_to_scene.py reconstruct outputs/route --overwrite
```

The default motion threshold is 0.5 m/s. Use
`--dynamic-min-speed-mps` on either `segment` or `track` to change it. The
model-free tracking pass retains multiple coherent range layers long enough to
prefer the one agreeing with a confirmed-static multi-frame track; this avoids
scaling a distant object from foreground LiDAR leakage.

Completed stages are skipped when their source and configuration hashes match.
A failed stage resumes from existing unit artifacts where possible. Changing a
completed stage's configuration requires `--overwrite`; that flag removes only
artifacts owned by that stage and its downstream stages.

## Outputs and coordinates

The run directory contains:

- `route-manifest.json`, `trajectory.csv`, and lossless trajectory/calibration
  arrays;
- lossless keyframe RGB, range, per-column timestamps, and per-column SLAM
  poses;
- each frame's standard `sam3-mask-manifest/v1` and PNG masks;
- `tracks.json`, all-observation point samples, range-qualified reconstruction
  points, selected RGB/mask pairs, and fit diagnostics, including the raw SAM
  3D pose and orientation-prior decisions;
- one `meshes/<track-id>.glb` per successful static track;
- `scene.glb` and an authoritative `scene.json` transform listing;
- optional unclassified binary RGB `route.ply` visual context.

All transformations use column-vector notation. A range pixel is placed as:

```text
p_world = T_world_body[column] @ T_body_sensor @ p_sensor
```

GLB geometry remains local and its root node carries the local-SLAM transform.
Coordinates are in metres and are not georeferenced.

For bounded OSF extraction, `source_window` in `route-manifest.json` records
the requested and effective bounds, recording length, processed count,
numbering convention, and fresh-SLAM origin behavior. Trajectory and keyframe
scan indices remain absolute positions in the source recording.

If a track has insufficient coherent range pixels, is dynamic, unconfirmed, or
outside the mesh range, or fails SAM 3D reconstruction, its reason remains in
`tracks.json`; other tracks continue processing.

## Prompted route surfaces as a TIN

Roads and other large, approximately 2.5D surfaces should not be passed to SAM
3D's bounded-object reconstruction. The surface workflow uses SAM 3 only to
identify registered range pixels, fuses those metric LiDAR returns, and builds
an open triangulated irregular network directly in the local SLAM frame.

Run the complete workflow with one or more literal surface descriptions:

```bash
sam3d-ouster-route surface run /data/route.osf \
  --output-dir outputs/road-surface \
  --prompt "dirt road" \
  --prompt "gravel carriageway"
```

Surface runs select a keyframe every 1 metre by default. The usual extraction
options, including OSF frame windows and `--meta` for PCAP recordings, remain
available. All predictions from all supplied prompts are unioned into one
surface; no prompt word or synonym is special-cased.

The workflow is also staged so SAM 3 does not need to run again while tuning
the TIN:

```bash
sam3d-ouster-route extract /data/route.osf \
  --output-dir outputs/road-surface \
  --keyframe-distance-m 1

sam3d-ouster-route surface segment outputs/road-surface \
  --prompt "unpaved track" \
  --prompt "gravel road"

sam3d-ouster-route surface build outputs/road-surface \
  --surface-resolution-m 0.20 \
  --max-surface-range-m 30 \
  --max-triangle-edge-m 1.0 \
  --max-slope-deg 45 \
  --tin-tile-size-m 50 \
  --fill-holes \
  --max-hole-width-m 1.0
```

Use `--no-max-surface-range` to retain all valid masked ranges. Reducing
`--surface-resolution-m` retains more detail but increases point count,
triangulation memory, and GLB size. The triangulation is tiled and uses local
point spacing, a hard edge limit, slope filtering, and centroid/edge-midpoint
support checks. Hole filling is disabled by default. With `--fill-holes`, a
second pass can restore complete enclosed rejected-face regions no wider than
`--max-hole-width-m` (1 metre by default). Regions touching the triangulation
exterior, spanning disconnected road components, or containing steep or
degenerate faces remain open, so the repair does not close the road shoulders.
When enabled, the tile size must exceed twice the sum of the triangle-edge and
hole-width limits so repaired gaps have adequate overlap at tile seams.

Surface-owned artifacts do not replace object masks, tracks, or scenes:

- `frames/<frame-id>/surface-segmentation/` contains the original SAM 3 masks;
- `surface/surface-points.ply` is the fused binary RGB point cloud used as the
  TIN vertex set;
- `surface/surface.glb` is one vertex-colored open mesh, including any
  disconnected supported components;
- `surface/surface.json` records prompts, configuration, bounds, frame and
  point counts, rejected-face reasons, and component areas.

Coordinates remain in metres in the recording window's local, non-georeferenced
SLAM frame. A TIN has one elevation per XY location, so overlapping decks such
as stacked roads or an overpass above another selected surface are not
supported in a single output.

For a quality check, view the PLY and GLB together and verify that the fused
points stop at the observed shoulders, face normals point upward, and no faces
span medians, intersections, or unobserved gaps. An opt-in end-to-end smoke test
is available when a small OSF fixture and the masking executable are present:

```bash
RUN_SAM3_SURFACE_INTEGRATION=1 \
SAM3_ROUTE_OSF_FIXTURE=/data/route.osf \
SAM3_MASK_ROUTE_EXECUTABLE=/home/ubuntu/micromamba/envs/sam3-masking/bin/sam3-mask-route \
SAM3_SURFACE_TEST_PROMPT="gravel road" \
pytest -m gpu tests/test_ouster_route_surface.py -s
```
