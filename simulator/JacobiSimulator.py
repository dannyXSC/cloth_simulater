from utils.ImplicitTool import *


@ti.kernel
def jacobiIteratrion():
    """
    雅可比迭代法 解稀疏矩阵的线性方程组
    A = M - J_x*dt^2 - J_v*dt
    b = force * dt + J_x * v * dt**2
    A*dv(t+1) = b
    """
    for i in range(N):
        r = b[i]
        for j in range(N):
            if i != j:
                r -= A[i, j] @ dv[j]

        new_v[i] = A[i, i].inverse() @ r

    for i in range(N):
        dv[i] = new_v[i]


def simulate(iter_times=10):
    initM()
    updateJX()
    updateJV()
    updateA()
    updateForce()
    updateB()
    # LSICS_updateB()
    initDv()
    for _ in range(iter_times):
        jacobiIteratrion()
    updateV()
    simpleCollision()
    # LSICS_simpleCollision()
    updateX()


cur_t = 0.0


def step(max_time=5):
    global cur_t
    if cur_t > max_time:
        initScene()
        updateVertices()
        cur_t = 0.0
    for _ in range(substeps):
        simulate()
        cur_t += dt

    updateVertices()

    scene = getScene()
    scene.mesh(vertices, indices=indices, per_vertex_color=colors, two_sided=True)
    return scene
