from utils.ImplicitTool import *


@ti.kernel
def GSIteratrion():
    """
    Gauss–Seidel迭代法 解稀疏矩阵的线性方程组
    """
    # 阻止taichi并行
    if 1:
        for i in range(N):
            r = b[i]
            for j in range(N):
                if i != j:
                    r -= A[i, j] @ dv[j]

            dv[i] = A[i, i].inverse() @ r


def simulate(iter_times=10):
    initM()
    updateJX()
    updateJV()
    updateA()
    updateForce()
    updateB()
    initDv()
    for _ in range(iter_times):
        GSIteratrion()
    updateV()
    simpleCollision()
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
