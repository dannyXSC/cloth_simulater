import taichi as ti
import taichi.math as tm

import math

ti.init(arch=ti.gpu)


# numbers of particles per row
n = 64
# total numbers of particles
N = n * n
# total mass
total_mass = 400 * 400
# mass of each partical
particle_mass = total_mass / N
# determin consider how many neighbors' force
bending_springs = False
# ks
spring_stiffness = 3e5
# kd
dashpot_damping = 2e4
# gravity
gravity = ti.Vector([0, -9.8, 0]) * 50

# epsilon distance
epsilon_distance = 1

# ball_radius = 0.3
ball_radius = 100
ball_center = ti.Vector.field(3, dtype=float, shape=(1,))
# ball_center[0] = [0, 0, 0]
ball_center[0] = [250, 100, 200]

cloth_width = 400
quad_size = cloth_width / n
num_triangles = (n - 1) * (n - 1) * 2
indices = ti.field(int, shape=num_triangles * 3)
vertices = ti.Vector.field(3, dtype=float, shape=N)
colors = ti.Vector.field(3, dtype=float, shape=N)

floor_vertices = ti.Vector.field(3, dtype=float, shape=4)
floor_indices = ti.field(int, shape=6)
floor_colors = ti.Vector.field(3, dtype=float, shape=4)
floor_vertices[0] = [-100.0, 0.0, -100.0]
floor_vertices[1] = [1000.0, 0.0, -100.0]
floor_vertices[2] = [1000.0, 0.0, 1000.0]
floor_vertices[3] = [-100.0, 0.0, 1000.0]
floor_indices[0] = 0
floor_indices[1] = 1
floor_indices[2] = 2
floor_indices[3] = 0
floor_indices[4] = 2
floor_indices[5] = 3
floor_colors[0] = (1.0, 1.0, 1.0)
floor_colors[1] = (1.0, 1.0, 1.0)
floor_colors[2] = (1.0, 1.0, 1.0)
floor_colors[3] = (1.0, 1.0, 1.0)

# time per step
dt = 0.04 / n
substeps = int(1 / 60 // dt)


"""
neighbors setting
"""
neighbor_offsets = []
if bending_springs:
    for i in range(-1, 2):
        for j in range(-1, 2):
            if (i, j) != (0, 0):
                neighbor_offsets.append([i, j, quad_size * math.sqrt(i * i + j * j)])
else:
    for i in range(-2, 3):
        for j in range(-2, 3):
            if (i, j) != (0, 0) and abs(i) + abs(j) <= 2:
                neighbor_offsets.append([i, j, quad_size * math.sqrt(i * i + j * j)])


@ti.kernel
def initVertices():
    # random_offset = ti.Vector([ti.random() - 0.5, ti.random() - 0.5])
    random_offset = ti.Vector([0, 0])

    for i in range(n):
        for j in range(n):
            idx = i * n + j
            # x[idx] = [
            #     i * quad_size - 0.5 + random_offset[0] * 0.1,
            #     0.6,
            #     j * quad_size - 0.5 + random_offset[1] * 0.1,
            # ]
            vertices[idx] = [
                ball_center[0][0]
                - cloth_width / 2
                + (i - 0.5 + random_offset[0] * 0.1) * quad_size,
                300,
                ball_center[0][2]
                - cloth_width / 2
                + (j - 0.5 + random_offset[1] * 0.1) * quad_size,
            ]


@ti.kernel
def initIndices():
    """
    init triangle
    """
    for i, j in ti.ndrange(n - 1, n - 1):
        idx = (i * (n - 1) + j) * 6
        indices[idx + 0] = i * n + j + 0
        indices[idx + 1] = (i + 1) * n + j
        indices[idx + 2] = i * n + j + 1

        indices[idx + 3] = i * n + j + 1
        indices[idx + 4] = (i + 1) * n + j
        indices[idx + 5] = (i + 1) * n + j + 1

    for i, j in ti.ndrange(n, n):
        if (i // 4 + j // 4) % 2 == 0:
            colors[i * n + j] = (0.22, 0.72, 0.52)
        else:
            colors[i * n + j] = (1, 0.334, 0.52)


scene = ti.ui.Scene()

camera = ti.ui.Camera()
# camera.position(0.0, 0.0, 3)
camera.position(278, 273, -800)
# camera.lookat(0,0,0)
camera.lookat(278, 273, 200)
camera.up(0, 1, 0)
camera.fov(40)
camera.z_near(10.0)
camera.z_far(1000)
scene.set_camera(camera)


def getScene() -> ti.ui.Scene:
    scene.point_light(pos=(500, 1000, 500), color=(1, 1, 1))
    scene.ambient_light((0.5, 0.5, 0.5))
    scene.particles(ball_center, radius=ball_radius, color=(0.5, 0.42, 0.8))
    scene.mesh(
        floor_vertices,
        indices=floor_indices,
        per_vertex_color=floor_colors,
        two_sided=True,
    )
    return scene


initIndices()
