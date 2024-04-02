from utils.SceneSetting import *

# const matrix
I = ti.Matrix.field(3, 3, dtype=float, shape=(1,))
I[0] = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
O = ti.Matrix.field(3, 3, dtype=float, shape=(1,))
O[0] = ti.Matrix([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
O_v = ti.Vector.field(3, dtype=float, shape=(1,))
O_v[0] = ti.Vector([0.0, 0.0, 0.0])

# position
x = ti.Vector.field(3, dtype=float, shape=N)
# speed
v = ti.Vector.field(3, dtype=float, shape=N)
# delta speed
dv = ti.Vector.field(3, dtype=float, shape=N)
# temp v
new_v = ti.Vector.field(3, dtype=float, shape=N)
# force
force = ti.Vector.field(3, dtype=float, shape=N)
# M
M = ti.Matrix.field(3, 3, dtype=float, shape=(N, N))
# The partial of the force with respect to x
J_x = ti.Matrix.field(3, 3, dtype=float, shape=(N, N))
# The partial of the force with respect to v
J_v = ti.Matrix.field(3, 3, dtype=float, shape=(N, N))
# A  for Av = b
A = ti.Matrix.field(3, 3, dtype=float, shape=(N, N))
# b  for Av = b
b = ti.Vector.field(3, dtype=float, shape=N)
# y is the position aleration term when collision happens
y = ti.Vector.field(3, dtype=float, shape=N)


@ti.kernel
def initDv():
    for i in range(N):
        for j in range(3):
            dv[i][j] = 0


@ti.kernel
def initX():
    for i in range(n):
        for j in range(n):
            idx = i * n + j
            x[idx] = vertices[idx]
            v[idx] = [0, 0, 0]


@ti.kernel
def updateV():
    for i in range(N):
        v[i] += dv[i]


@ti.kernel
def updateX():
    for i in range(N):
        x[i] += v[i] * dt + y[i]


@ti.kernel
def updateVertices():
    for i in range(N):
        vertices[i] = x[i]


@ti.kernel
def initM():
    """
    初始化质量矩阵
    """
    m = ti.Matrix([[particle_mass, 0, 0], [0, particle_mass, 0], [0, 0, particle_mass]])

    for i in range(N):
        M[i, i] = m


@ti.kernel
def updateJX():
    """
    calculate the partial of the force with respect to x
    """
    for i, j in J_x:
        J_x[i, j] = O[0]

    for i in range(N):
        # i is the subject to force
        # just calculate the J_x on i
        for off_r, off_c, d_orign in ti.static(neighbor_offsets):
            if (0 <= i // n + off_r < n) and (0 <= i % n + off_c < n):
                # neighbor is the particle that exerts the force
                neighbor = i + off_r * n + off_c

                x_ij = x[i] - x[neighbor]
                d = ti.math.length(x_ij)
                x_ij_norm = ti.math.normalize(x_ij)
                # x_ij_norm * x_ij_norm T
                m = x_ij_norm.outer_product(x_ij_norm)
                # J_x_ij = \partial force_i / \partial x_j
                J_x[i, neighbor] = spring_stiffness * (
                    (1 - d_orign / d) * (I[0] - m) + m
                )

                J_x[i, i] -= J_x[i, neighbor]


@ti.kernel
def updateJV():
    """
    calculate the partial of the force with respect to v
    """
    for i, j in J_v:
        J_v[i, j] = O[0]

    for i in range(N):
        # i is the subject to force
        # just calculate the J_x on i
        for off_r, off_c, d_orign in ti.static(neighbor_offsets):
            if (0 <= i // n + off_r < n) and (0 <= i % n + off_c < n):
                # neighbor is the particle that exerts the force
                neighbor = i + off_r * n + off_c

                x_ij = x[i] - x[neighbor]
                x_ij_norm = ti.math.normalize(x_ij)

                J_v[i, neighbor] = dashpot_damping * x_ij_norm.outer_product(x_ij_norm)

                J_v[i, i] -= J_v[i, neighbor]


@ti.kernel
def updateA():
    """
    update A
    A = M - J_x*dt^2 - J_v*dt
    """
    for i, j in A:
        A[i, j] = M[i, j] - J_x[i, j] * dt * dt - J_v[i, j] * dt


@ti.kernel
def updateForce():
    """
    update force
    """
    for i in range(N):
        force[i] = particle_mass * gravity

    for i in range(N):
        # i is the subject to force
        for off_r, off_c, d_orign in ti.static(neighbor_offsets):
            if (0 <= i // n + off_r < n) and (0 <= i % n + off_c < n):
                # neighbor is the particle that exerts the force
                neighbor = i + off_r * n + off_c

                x_ij = x[i] - x[neighbor]
                v_ij = v[i] - v[neighbor]
                d = ti.math.length(x_ij)
                x_ij_norm = ti.math.normalize(x_ij)

                # spring force
                force[i] += -spring_stiffness * x_ij_norm * (d - d_orign)
                # damping force
                force[i] += -dashpot_damping * x_ij_norm * (v_ij.dot(x_ij_norm))


@ti.kernel
def updateB():
    """
    update b
    b = force * dt + J_x * v * dt**2
    """
    for i in range(N):
        tmp = ti.Vector([0.0, 0.0, 0.0])
        for off_r, off_c, d_orign in ti.static(neighbor_offsets):
            if (0 <= i // n + off_r < n) and (0 <= i % n + off_c < n):
                # neighbor is the particle that exerts the force
                neighbor = i + off_r * n + off_c

                tmp += J_x[i, neighbor] @ v[neighbor]
        tmp += J_x[i, i] @ v[i]
        b[i] = force[i] * dt + tmp * dt * dt


@ti.kernel
def LSICS_updateB():
    """
    update b
    b = (force + J_x * (v * dt + y) )* dt
    """
    for i in range(N):
        term = force[i]
        for off_r, off_c, d_orign in ti.static(neighbor_offsets):
            if (0 <= i // n + off_r < n) and (0 <= i % n + off_c < n):
                # neighbor is the particle that exerts the force
                neighbor = i + off_r * n + off_c

                term += J_x[i, neighbor] @ (v[neighbor] * dt + y[neighbor])
        term += J_x[i, i] @ (v[i] * dt + y[i])
        b[i] = term * dt


@ti.kernel
def simpleCollision():
    """
    detect collision with object and floor
    """
    for i in range(N):
        offset_to_center = x[i] - ball_center[0]
        distance = offset_to_center.norm()
        if distance <= ball_radius / 0.9:
            normal = ti.math.normalize(offset_to_center)
            dot_result = v[i].dot(normal)
            # 如果速度朝向圆心，那么给予反向速度
            # v[i] -= 2 * min(dot_result, 0) * normal
            v[i] -= min(dot_result, 0) * normal
        # if x[i][1] <= epsilon_distance:
        #     v[i][1] = 0


@ti.kernel
def LSICS_simpleCollision():
    """
    detect collision with object and floor
    and use LSICS method to change the position of the particle
    """
    for i in range(N):
        flag = False

        offset_to_center = x[i] - ball_center[0]
        distance = offset_to_center.norm()
        if distance <= ball_radius + epsilon_distance:
            normal = ti.math.normalize(offset_to_center)
            dot_result = v[i].dot(normal)
            # 如果速度朝向圆心，那么给予反向速度
            # v[i] -= 2 * min(dot_result, 0) * normal
            v[i] -= min(dot_result, 0) * normal
            y[i] = normal * (ball_radius - distance + epsilon_distance)
            flag = True
        if x[i][1] <= 0:
            v[i][1] = 0
            y[i] = [0, -x[i][1] + epsilon_distance, 0]

            flag = True
        if not flag:
            y[i] = ti.Vector([0.0, 0.0, 0.0])


def initScene():
    initVertices()
    initX()


initScene()
