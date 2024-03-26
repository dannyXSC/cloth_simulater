import taichi as ti
import taichi.math as tm

from simulator.JacobiSimulator import step

window = ti.ui.Window("Taichi Cloth Simulation on GGUI", (1024, 1024), vsync=True)
canvas = window.get_canvas()
canvas.set_background_color((0, 0, 0))

while window.running:
    scene = step(100)
    canvas.scene(scene)
    window.show()
