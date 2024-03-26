from utils.CGTool import *


def implicit_euler():
    """
    隐式欧拉求解
    """
    initM()
    updateJX()
    updateJV()
    updateA()
    updateForce()
    updateB()
    initDv()
    updateP()
    CGMethod()
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
        implicit_euler()
        cur_t += dt

    updateVertices()

    scene = getScene()
    scene.mesh(vertices, indices=indices, per_vertex_color=colors, two_sided=True)
    return scene
