"""
VTOL Motor Arm — Neo-Classical Render Script
Blender 5.1.1 | Setup-only — no import, no deletion, no operator context needed.

WORKFLOW:
    1. In 3D Viewport: A → X → Delete  (clear default objects)
    2. File → Import → Wavefront OBJ → select VTOL_Motor_Arm.obj → Import OBJ
    3. Scripting tab → open this file → Run Script  (Alt+P)
    4. Press F12 to render
    Output: C:\\Users\\HP\\Downloads\\vtol_arm_neoclassical.png
"""

import bpy
import math
import mathutils

OUTPUT_PATH = r"C:\Users\HP\Downloads\vtol_arm_neoclassical.png"

# ─── Find the imported arm ────────────────────────────────────────────────────
arm = next((o for o in bpy.data.objects if o.type == 'MESH'), None)
if arm is None:
    raise RuntimeError("No mesh in scene. Import the OBJ first via File → Import → Wavefront OBJ.")

print(f"[found] Mesh object: '{arm.name}'")
arm.name = "VTOL_Motor_Arm"


# ─── Scale correction: mm → m if needed ──────────────────────────────────────
verts = arm.data.vertices
xs = [v.co.x for v in verts]
ys = [v.co.y for v in verts]
zs = [v.co.z for v in verts]
bbox_max = max(max(xs)-min(xs), max(ys)-min(ys), max(zs)-min(zs))

if bbox_max > 1.0:
    arm.data.transform(mathutils.Matrix.Scale(0.001, 4))
    print(f"[scale] mm→m applied (was {bbox_max:.1f} units)")
    xs = [v.co.x for v in arm.data.vertices]
    ys = [v.co.y for v in arm.data.vertices]
    zs = [v.co.z for v in arm.data.vertices]

arm_length = max(xs) - min(xs)
arm_width  = max(ys) - min(ys)
arm_height = max(zs) - min(zs)
print(f"[geom] {arm_length*1000:.1f} × {arm_width*1000:.1f} × {arm_height*1000:.1f} mm")


# ─── Center to origin ─────────────────────────────────────────────────────────
cx = (max(xs) + min(xs)) / 2
cy = (max(ys) + min(ys)) / 2
cz = (max(zs) + min(zs)) / 2
arm.data.transform(mathutils.Matrix.Translation((-cx, -cy, -cz)))
arm.location = (0.0, 0.0, 0.0)


# ─── Smooth shading — polygon-level, no operator context needed ───────────────
for poly in arm.data.polygons:
    poly.use_smooth = True
arm.data.update()

# Edge Split preserves sharp corners (>55°) while keeping curved surfaces smooth
if "EdgeSplit" not in arm.modifiers:
    es = arm.modifiers.new("EdgeSplit", 'EDGE_SPLIT')
    es.split_angle    = math.radians(55)
    es.use_edge_angle = True
    es.use_edge_sharp = True
print("[smooth] Smooth shading + Edge Split applied")


# ─── Remove existing cameras and lights (from default scene or previous run) ──
for obj in list(bpy.data.objects):
    if obj.type in {'CAMERA', 'LIGHT'}:
        bpy.data.objects.remove(obj, do_unlink=True)
for cam in list(bpy.data.cameras):
    bpy.data.cameras.remove(cam)
for light in list(bpy.data.lights):
    bpy.data.lights.remove(light)


# ─── Material: matte dark aluminum, Neo-Classical palette ────────────────────
for mat in list(bpy.data.materials):
    bpy.data.materials.remove(mat)

mat = bpy.data.materials.new("NeoClassical_DarkAluminum")
mat.use_nodes = True
nt = mat.node_tree
nt.nodes.clear()

bsdf = nt.nodes.new('ShaderNodeBsdfPrincipled')
bsdf.location = (0, 0)
bsdf.inputs['Base Color'].default_value = (0.015, 0.015, 0.015, 1.0)  # #0a0a0a
bsdf.inputs['Metallic'].default_value   = 0.92
bsdf.inputs['Roughness'].default_value  = 0.28

mat_out = nt.nodes.new('ShaderNodeOutputMaterial')
mat_out.location = (320, 0)
nt.links.new(bsdf.outputs['BSDF'], mat_out.inputs['Surface'])

arm.data.materials.clear()
arm.data.materials.append(mat)
print("[material] NeoClassical dark aluminum assigned")


# ─── World: near-black void ───────────────────────────────────────────────────
world = bpy.context.scene.world
world.use_nodes = True
bg = world.node_tree.nodes.get('Background')
if bg:
    bg.inputs['Color'].default_value    = (0.004, 0.004, 0.004, 1.0)
    bg.inputs['Strength'].default_value = 0.0


# ─── Camera: 85mm telephoto, diagonal hero composition ───────────────────────
cam_data = bpy.data.cameras.new("Neo_Camera")
cam_data.lens = 85
cam_obj = bpy.data.objects.new("Neo_Camera", cam_data)
bpy.context.scene.collection.objects.link(cam_obj)
bpy.context.scene.camera = cam_obj

d = arm_length * 1.8
cam_obj.location = (arm_length * 0.25, -d * 0.75, d * 0.45)

track = cam_obj.constraints.new(type='TRACK_TO')
track.target     = arm
track.track_axis = 'TRACK_NEGATIVE_Z'
track.up_axis    = 'UP_Y'
print("[camera] 85mm telephoto — tracking arm center")


# ─── Lighting: 3-point Neo-Classical ─────────────────────────────────────────
def add_light(name, ltype, loc, energy, color, **kw):
    ld = bpy.data.lights.new(name, type=ltype)
    lo = bpy.data.objects.new(name, ld)
    bpy.context.scene.collection.objects.link(lo)
    lo.location = loc
    ld.energy   = energy
    ld.color    = color
    for k, v in kw.items():
        try: setattr(ld, k, v)
        except AttributeError: pass
    tc = lo.constraints.new('TRACK_TO')
    tc.target = arm; tc.track_axis = 'TRACK_NEGATIVE_Z'; tc.up_axis = 'UP_Y'

# Key — hard spot, upper-left, warm white, casts sharp shadows
add_light("Key_Light", 'SPOT',
    loc=(-arm_length*2.2, -arm_length*1.5, arm_length*3.5),
    energy=1400, color=(1.0, 0.97, 0.93),
    spot_size=math.radians(30), spot_blend=0.06, shadow_soft_size=0.004)

# Fill — large soft area, lower-right, cool blue at 10% key strength
add_light("Fill_Light", 'AREA',
    loc=(arm_length*1.8, arm_length*1.2, -arm_length*0.6),
    energy=180, color=(0.82, 0.88, 1.0),
    size=arm_length*3.5)

# Rim — behind arm, catches exterior fillet edges
add_light("Rim_Light", 'SPOT',
    loc=(arm_length*0.6, arm_length*2.2, arm_length*1.2),
    energy=550, color=(1.0, 1.0, 1.0),
    spot_size=math.radians(42), spot_blend=0.18)

print("[lighting] Key + Fill + Rim configured")


# ─── Render settings: Cycles 256spp, 1920×1080 PNG ───────────────────────────
sc = bpy.context.scene
sc.render.engine                     = 'CYCLES'
sc.cycles.samples                    = 256
sc.cycles.use_denoising              = True
sc.render.resolution_x               = 1920
sc.render.resolution_y               = 1080
sc.render.resolution_percentage      = 100
sc.render.image_settings.file_format = 'PNG'
sc.render.filepath                   = OUTPUT_PATH

try:
    prefs = bpy.context.preferences.addons['cycles'].preferences
    prefs.compute_device_type = 'CUDA'
    prefs.get_devices()
    for dev in prefs.devices:
        dev.use = True
    sc.cycles.device = 'GPU'
    print("[render] GPU (CUDA) enabled")
except Exception:
    sc.cycles.device = 'CPU'
    print("[render] CPU mode")

bpy.context.view_layer.update()

print("\n══════════════════════════════════════════════════════")
print("  Neo-Classical render setup complete")
print(f"  Arm: {arm_length*1000:.0f}mm × {arm_width*1000:.0f}mm × {arm_height*1000:.0f}mm")
print(f"  Output → {OUTPUT_PATH}")
print("  Press F12 to render")
print("══════════════════════════════════════════════════════\n")
