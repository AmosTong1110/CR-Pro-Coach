"""Blender-side renderer for CRbot sprites (runs inside Blender, headless). Driven by tools/art/build_sprites.py.

    blender -b --factory-startup --disable-autoexec --python tools/art/blender_render.py -- JOB.json

Job kinds ("mode"):
  char    animated character -> PNG frames per side (blue / red tint on cloth parts), facing (up / down), action
  static  one building model -> PNG (towers, castles)
  arena   the whole arena background (procedural grass / stone / dirt / water, own plank bridges, CC0 scenery)

World units = arena tiles. Orthographic cameras tilted from straight down: the ground at TILT, characters at
CHAR_TILT and buildings at STATIC_TILT (more front-facing so faces and roofs read). A ground point (x, y) lands on
screen at (x, y * cos(TILT)), which the board in frontend_app/qt/board.py uses for its tilted mapping.
Models: CC0 KayKit packs by Kay Lousberg (see CREDITS.md); nothing from any commercial game.
"""
import json
import math
import os
import random
import sys

import bpy
from mathutils import Vector

TILT = 35.0
CHAR_TILT = 58.0
STATIC_TILT = 48.0
PX_PER_TILE = 64
W, H = 18.0, 32.0
MARGIN_X, MARGIN_Y = 2.0, 2.2
RIVER_Y0, RIVER_Y1 = 15.0, 17.0
BRIDGE_X = (3.5, 14.5)


# -- scene helpers -----------------------------------------------------------------------------------------------
def reset():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    sc = bpy.context.scene
    for eng in ("BLENDER_EEVEE_NEXT", "BLENDER_EEVEE"):
        try:
            sc.render.engine = eng
            break
        except TypeError:
            continue
    sc.render.film_transparent = True
    sc.render.image_settings.file_format = "PNG"
    sc.render.image_settings.color_mode = "RGBA"
    sc.render.image_settings.compression = 90
    sc.view_settings.view_transform = "Standard"
    sc.view_settings.look = "None"
    try:
        sc.eevee.taa_render_samples = 16
    except AttributeError:
        pass
    world = bpy.data.worlds.new("w")
    sc.world = world
    world.use_nodes = True
    bg = world.node_tree.nodes["Background"]
    bg.inputs[0].default_value = (0.62, 0.72, 0.9, 1)
    bg.inputs[1].default_value = 0.55
    sun = bpy.data.objects.new("sun", bpy.data.lights.new("sun", "SUN"))
    sun.data.energy = 4.5
    sun.data.angle = math.radians(3)
    sun.rotation_euler = (math.radians(40), math.radians(-25), math.radians(-30))
    sc.collection.objects.link(sun)
    return sc


def camera(sc, cx, cy, ortho, res_x, res_y, tilt):
    """Orthographic camera looking at (cx, cy, 0); `ortho` spans the LONGER image side (Blender's rule)."""
    cam = bpy.data.objects.new("cam", bpy.data.cameras.new("cam"))
    cam.data.type = "ORTHO"
    cam.data.ortho_scale = ortho
    t = math.radians(tilt)
    d = 60.0
    cam.location = (cx, cy - d * math.sin(t), d * math.cos(t))
    cam.rotation_euler = (t, 0, 0)
    cam.data.clip_end = 500
    sc.collection.objects.link(cam)
    sc.camera = cam
    sc.render.resolution_x, sc.render.resolution_y = res_x, res_y
    sc.render.resolution_percentage = 100
    return cam


def import_model(path):
    before = set(bpy.data.objects)
    bpy.ops.import_scene.gltf(filepath=path)
    return [o for o in bpy.data.objects if o not in before]


def bbox(objs):
    pts = [o.matrix_world @ Vector(c) for o in objs if o.type == "MESH" and not o.hide_render for c in o.bound_box]
    lo = Vector((min(p.x for p in pts), min(p.y for p in pts), min(p.z for p in pts)))
    hi = Vector((max(p.x for p in pts), max(p.y for p in pts), max(p.z for p in pts)))
    return lo, hi


def roots(objs):
    return [o for o in objs if o.parent is None]


def place(objs, x, y, scale=1.0, rot_z=0.0):
    for r in roots(objs):
        r.rotation_mode = "XYZ"
        r.scale = (r.scale[0] * scale, r.scale[1] * scale, r.scale[2] * scale)
        r.location = (x, y, 0)
        r.rotation_euler[2] += rot_z
    bpy.context.view_layer.update()


def render_to(sc, path):
    sc.render.filepath = path
    bpy.ops.render.render(write_still=True)


# -- team tint ---------------------------------------------------------------------------------------------------
def tint_parts(objs, tokens):
    """Give cloth parts (capes, cloaks, hoods) their own material with a COLOR-blend node; returns the colour inputs."""
    inputs = []
    for o in objs:
        if o.type != "MESH" or o.hide_render or not any(t.lower() in o.name.lower() for t in tokens):
            continue
        for slot in o.material_slots:
            if slot.material is None or not slot.material.use_nodes:
                continue
            m = slot.material.copy()
            slot.material = m
            nt = m.node_tree
            bsdf = next((n for n in nt.nodes if n.type == "BSDF_PRINCIPLED"), None)
            if bsdf is None:
                continue
            base = bsdf.inputs["Base Color"]
            src = base.links[0].from_socket if base.links else None
            mix = nt.nodes.new("ShaderNodeMix")
            mix.data_type = "RGBA"
            mix.blend_type = "COLOR"                    # keeps the shading (value), takes hue + saturation
            mix.inputs["Factor"].default_value = 0.9
            if src is not None:
                nt.links.new(src, mix.inputs[6])
            else:
                mix.inputs[6].default_value = base.default_value
            nt.links.new(mix.outputs[2], base)
            inputs.append(mix.inputs[7])
    return inputs


# -- jobs --------------------------------------------------------------------------------------------------------
def job_char(job):
    sc = reset()
    objs = import_model(job["model"])
    keep = job.get("keep")
    for o in objs:
        if o.type == "MESH" and (o.name == "Icosphere" or (keep is not None and o.name not in keep)):
            o.hide_render = True
    arm = next((o for o in objs if o.type == "ARMATURE"), None)
    for path2, bone in job.get("attach", []):        # weapon props parented to a hand-slot bone
        extra = import_model(path2)
        for r in roots(extra):
            r.parent = arm
            r.parent_type = "BONE"
            r.parent_bone = bone
            r.location = (0, 0, 0)
            r.matrix_parent_inverse.identity()
        objs += extra
    bpy.context.view_layer.update()
    lo, hi = bbox(objs)
    place(objs, 0, 0, job.get("height", 2.0) / max(hi.z - lo.z, 1e-6))
    tint = tint_parts(objs, job.get("tint_parts", []))
    frame_tiles = job.get("frame_tiles", 3.0)
    res = int(frame_tiles * PX_PER_TILE)
    camera(sc, 0, 0, frame_tiles, res, res, CHAR_TILT)   # feet (origin) at the image centre...
    sc.camera.data.shift_y = 0.20                        # ...moved to 70 % of the image height
    out = job["out_dir"]
    os.makedirs(out, exist_ok=True)
    frames = {}
    for side, rgb in job["sides"].items():
        for sock in tint:
            sock.default_value = (*rgb, 1)
        for facing, rz in (("down", 0.0), ("up", math.pi)):
            if arm is not None:
                arm.rotation_mode = "XYZ"
                arm.rotation_euler[2] = rz
            for label, (action, n) in job["actions"].items():
                act = bpy.data.actions.get(action)
                if arm is None or act is None:
                    print("MISSING action", action)
                    continue
                if arm.animation_data is None:
                    arm.animation_data_create()
                arm.animation_data.action = act
                f0, f1 = act.frame_range
                files = []
                for i in range(n):
                    span = i / max(1, n - 1) if label == "death" else i / n   # death ends on its last pose
                    sc.frame_set(int(round(f0 + (f1 - f0) * span)))
                    fn = f"{job['name']}_{side}_{facing}_{label}_{i}.png"
                    render_to(sc, os.path.join(out, fn))
                    files.append(fn)
                frames[f"{side}/{facing}/{label}"] = files
    with open(os.path.join(out, f"{job['name']}.json"), "w") as fh:
        json.dump({"name": job["name"], "cell": res, "feet": [0.5, 0.70], "px_per_tile": PX_PER_TILE,
                   "frames": frames}, fh, indent=1)


def job_static(job):
    sc = reset()
    objs = import_model(job["model"])
    lo, hi = bbox(objs)
    place(objs, 0, 0, job["footprint"] / max(hi.x - lo.x, hi.y - lo.y, 1e-6))
    for r in roots(objs):
        r.scale[2] *= job.get("squash", 1.0)
    bpy.context.view_layer.update()
    lo, hi = bbox(objs)
    t = math.radians(STATIC_TILT)
    size = max(job["footprint"] * 1.3, (job["footprint"] * math.cos(t) + (hi.z - lo.z) * math.sin(t)) * 1.15)
    res = int(size * PX_PER_TILE)
    camera(sc, 0, 0, size, res, res, STATIC_TILT)
    sc.camera.data.shift_y = ((hi.z - lo.z) * math.sin(t) * 0.5) / size
    render_to(sc, job["out"])
    with open(os.path.splitext(job["out"])[0] + ".json", "w") as fh:
        json.dump({"size": res, "px_per_tile": PX_PER_TILE, "ground_center": [0.5, 0.5 + sc.camera.data.shift_y],
                   "height_tiles": (hi.z - lo.z)}, fh)


def mat(name, rgb, rough=0.9):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    b = m.node_tree.nodes["Principled BSDF"]
    b.inputs["Base Color"].default_value = (*rgb, 1)
    b.inputs["Roughness"].default_value = rough
    return m


def _noise_ramp(nt, scale, detail, c0, c1, coord):
    noise = nt.nodes.new("ShaderNodeTexNoise")
    noise.inputs["Scale"].default_value = scale
    noise.inputs["Detail"].default_value = detail
    nt.links.new(coord, noise.inputs["Vector"])
    ramp = nt.nodes.new("ShaderNodeValToRGB")
    ramp.color_ramp.elements[0].position = 0.35
    ramp.color_ramp.elements[1].position = 0.7
    ramp.color_ramp.elements[0].color = (*c0, 1)
    ramp.color_ramp.elements[1].color = (*c1, 1)
    nt.links.new(noise.outputs["Fac"], ramp.inputs["Fac"])
    return ramp


def _node_mat(name, rough=0.95):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    nt = m.node_tree
    b = nt.nodes["Principled BSDF"]
    b.inputs["Roughness"].default_value = rough
    return m, nt, b, nt.nodes.new("ShaderNodeTexCoord").outputs["Object"]


def grass_mat():
    m, nt, b, coord = _node_mat("grass")
    big = _noise_ramp(nt, 1.2, 4, (0.13, 0.30, 0.06), (0.26, 0.48, 0.10), coord)
    fine = _noise_ramp(nt, 40.0, 2, (0.6, 0.6, 0.6), (1.0, 1.0, 1.0), coord)
    mix = nt.nodes.new("ShaderNodeMix")
    mix.data_type = "RGBA"
    mix.blend_type = "MULTIPLY"
    mix.inputs["Factor"].default_value = 0.6
    nt.links.new(big.outputs["Color"], mix.inputs[6])
    nt.links.new(fine.outputs["Color"], mix.inputs[7])
    nt.links.new(mix.outputs[2], b.inputs["Base Color"])
    return m


def stone_mat(name, c1, c2, mortar, scale):
    m, nt, b, coord = _node_mat(name, 0.85)
    br = nt.nodes.new("ShaderNodeTexBrick")
    br.inputs["Scale"].default_value = scale
    br.inputs["Color1"].default_value = (*c1, 1)
    br.inputs["Color2"].default_value = (*c2, 1)
    br.inputs["Mortar"].default_value = (*mortar, 1)
    br.inputs["Mortar Size"].default_value = 0.025
    br.inputs["Row Height"].default_value = 0.5
    br.inputs["Brick Width"].default_value = 0.5
    br.offset = 0.5
    nt.links.new(coord, br.inputs["Vector"])
    nt.links.new(br.outputs["Color"], b.inputs["Base Color"])
    bump = nt.nodes.new("ShaderNodeBump")
    bump.inputs["Strength"].default_value = 0.35
    nt.links.new(br.outputs["Fac"], bump.inputs["Height"])
    nt.links.new(bump.outputs["Normal"], b.inputs["Normal"])
    return m


def noise_mat(name, scale, detail, c0, c1, rough=1.0):
    m, nt, b, coord = _node_mat(name, rough)
    r = _noise_ramp(nt, scale, detail, c0, c1, coord)
    nt.links.new(r.outputs["Color"], b.inputs["Base Color"])
    return m


def plane(name, x0, y0, x1, y1, z, material):
    bpy.ops.mesh.primitive_plane_add(size=1, location=((x0 + x1) / 2, (y0 + y1) / 2, z))
    o = bpy.context.active_object
    o.name = name
    o.scale = (x1 - x0, y1 - y0, 1)
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)   # textures in world units
    o.data.materials.append(material)
    return o


def box(x, y, z, sx, sy, sz, material):
    bpy.ops.mesh.primitive_cube_add(size=1, location=(x, y, z))
    o = bpy.context.active_object
    o.scale = (sx, sy, sz)
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    o.data.materials.append(material)
    return o


def job_arena(job):
    sc = reset()
    rnd = random.Random(7)                            # fixed seed: the same arena every build
    plane("ground", -MARGIN_X, -MARGIN_Y, W + MARGIN_X, H + MARGIN_Y, 0, grass_mat())
    dirt = noise_mat("dirt", 6.0, 6, (0.36, 0.25, 0.14), (0.55, 0.42, 0.26))
    pad = stone_mat("pad", (0.62, 0.55, 0.42), (0.52, 0.45, 0.34), (0.30, 0.26, 0.20), 1.0)
    plane("pad", 1.0, 0.3, 17.0, 8.6, 0.004, pad)
    plane("pad2", 1.0, H - 8.6, 17.0, H - 0.3, 0.004, pad)
    for bx in BRIDGE_X:
        plane("lane", bx - 0.9, 2.0, bx + 0.9, H - 2.0, 0.006, dirt)
    plane("bank", -MARGIN_X, RIVER_Y0 - 0.25, W + MARGIN_X, RIVER_Y1 + 0.25, 0.01,
          stone_mat("bank", (0.42, 0.40, 0.36), (0.33, 0.31, 0.28), (0.2, 0.19, 0.17), 1.6))
    plane("water", -MARGIN_X, RIVER_Y0, W + MARGIN_X, RIVER_Y1, 0.02,
          noise_mat("water", 2.5, 3, (0.03, 0.20, 0.50), (0.10, 0.40, 0.72), 0.12))
    wood = noise_mat("wood", 6.0, 6, (0.36, 0.25, 0.14), (0.55, 0.42, 0.26))
    plank_a, plank_b = mat("plank_a", (0.42, 0.26, 0.12), 0.8), mat("plank_b", (0.50, 0.32, 0.15), 0.8)
    rail = mat("rail", (0.28, 0.17, 0.08), 0.8)
    cy = (RIVER_Y0 + RIVER_Y1) / 2
    for bx in BRIDGE_X:                               # own plank bridges
        box(bx, cy, 0.06, 2.3, 3.4, 0.12, wood)
        for i in range(10):
            box(bx, cy - 1.6 + i * 3.2 / 9, 0.15, 2.2, 0.3, 0.06, plank_a if i % 2 else plank_b)
        for sx in (-1, 1):
            box(bx + sx * 1.12, cy, 0.32, 0.14, 3.5, 0.12, rail)
            for py in (-1.6, 0, 1.6):
                box(bx + sx * 1.12, cy + py, 0.2, 0.2, 0.2, 0.4, rail)
    nat = os.path.join(job["root"], "decoration/nature")
    trees = ["tree_single_A.gltf", "tree_single_B.gltf", "trees_A_small.gltf", "trees_B_small.gltf"]
    rocks = ["rock_single_A.gltf", "rock_single_B.gltf", "rock_single_C.gltf"]
    for side_x in (-MARGIN_X / 2 - 0.2, W + MARGIN_X / 2 + 0.2):
        y = -MARGIN_Y + 0.5
        while y < H + MARGIN_Y - 0.5:
            if not (RIVER_Y0 - 0.6 < y < RIVER_Y1 + 0.6):
                objs = import_model(os.path.join(nat, rnd.choice(trees if rnd.random() < 0.7 else rocks)))
                lo, hi = bbox(objs)
                s = rnd.uniform(1.4, 2.0) / max(hi.x - lo.x, hi.y - lo.y, 1e-6)
                place(objs, side_x + rnd.uniform(-0.4, 0.4), y, s, rnd.uniform(0, 6.28))
            y += rnd.uniform(1.6, 2.4)
    t = math.radians(TILT)
    full_w, full_h = W + 2 * MARGIN_X, (H + 2 * MARGIN_Y) * math.cos(t)
    res_x, res_y = int(full_w * PX_PER_TILE), int(full_h * PX_PER_TILE)
    camera(sc, W / 2, H / 2, max(full_w, full_h), res_x, res_y, TILT)
    sc.render.film_transparent = False
    render_to(sc, job["out"])
    with open(os.path.splitext(job["out"])[0] + ".json", "w") as fh:
        json.dump({"px_per_tile": PX_PER_TILE, "tilt": TILT, "margin": [MARGIN_X, MARGIN_Y],
                   "size": [res_x, res_y]}, fh)


def job_inspect(job):
    reset()
    objs = import_model(job["model"])
    print("INSPECT", json.dumps({"objects": [(o.name, o.type) for o in objs],
                                 "actions": [a.name for a in bpy.data.actions]}))


if __name__ == "__main__":
    with open(sys.argv[sys.argv.index("--") + 1]) as fh:
        spec = json.load(fh)
    {"char": job_char, "static": job_static, "arena": job_arena, "inspect": job_inspect}[spec["mode"]](spec)
