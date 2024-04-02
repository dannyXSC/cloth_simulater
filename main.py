import taichi as ti
import taichi.math as tm

from simulator.JacobiSimulator import step as JacobiStep
from simulator.GSSimulator import step as GsStep
from simulator.CGSimulator import step as CgStep

window = ti.ui.Window("Taichi Cloth Simulation on GGUI", (1024, 1024), vsync=True)
canvas = window.get_canvas()
canvas.set_background_color((0, 0, 0))


def runWithJacobi():
    while window.running:
        scene = JacobiStep()
        canvas.scene(scene)
        window.show()


def runWithGs():
    while window.running:
        scene = GsStep()
        canvas.scene(scene)
        window.show()


def runWithCg():
    while window.running:
        scene = CgStep()
        canvas.scene(scene)
        window.show()


if __name__ == "__main__":
    # runWithJacobi()
    # runWithGs()
    runWithCg()
