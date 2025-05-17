import trimesh
import pyrender
import numpy as np
import imageio

# Load GLB model
mesh = trimesh.load('your_model.glb')

# Create scene and add mesh
scene = pyrender.Scene(ambient_light=[0.5, 0.5, 0.5])

for geo in mesh.geometry.values():
    scene.add(pyrender.Mesh.from_trimesh(geo))

# Set up camera
camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)
camera_pose = np.eye(4)
camera_pose[:3, 3] = [0, 0, 3]
scene.add(camera, pose=camera_pose)

# Optional: Add light if model appears too dark
light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=3.0)
scene.add(light, pose=camera_pose)

# Renderer with offscreen mode - RGBA output for transparency
r = pyrender.OffscreenRenderer(viewport_width=640, viewport_height=480)
r.render_flags = pyrender.constants.RenderFlags.RGBA

# Store frames
frames = []

# Rotate around Y-axis
nodes = [node for node in scene.nodes if node.mesh is not None]

for i in range(60):  # 60 frames
    angle = i * 0.1  # adjust rotation speed
    rot_matrix = trimesh.transformations.rotation_matrix(angle, [0, 1, 0])

    for node in nodes:
        node.matrix = rot_matrix @ np.eye(4)

    color, _ = r.render(scene)

    # Convert to uint8
    frame = (color * 255).astype(np.uint8)

    # Append to frames
    frames.append(frame)

# Save as GIF with transparency
imageio.mimsave('rotating_model_transparent.gif', frames, format='gif', fps=20, loop=0, transparency=0)

# Optionally, save first frame as PNG with alpha
imageio.imwrite('first_frame.png', frames[0])

print("✅ PNG & GIF saved with transparent background!")
