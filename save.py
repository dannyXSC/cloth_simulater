import taichi as ti
import taichi.math as tm

import os

from utils.SceneSetting import vertices, indices, N, num_triangles, n


def saveVedioMesh(result_dir, step):
    window = ti.ui.Window(
        "Taichi Cloth Simulation on GGUI", (1024, 1024), vsync=True, show_window=False
    )
    canvas = window.get_canvas()
    canvas.set_background_color((0, 0, 0))

    video_manager = ti.tools.VideoManager(
        output_dir=result_dir, framerate=60, automatic_build=False
    )
    mesh_path = os.path.join(result_dir, "mesh")
    os.makedirs(mesh_path, exist_ok=True)

    def saveMesh(idx):
        global vertices, indices, N, n
        with open(os.path.join(mesh_path, "{:0>6d}.obj".format(idx)), "w") as f:
            f.write("# OBJ file\n")
            for i in range(N):
                f.write(
                    "v {:.4f} {:.4f} {:.4f}\n".format(
                        vertices[i][0], vertices[i][1], vertices[i][2]
                    )
                )
            for i in range(N):
                r = (i // n) / n
                c = (i % n) / n
                f.write("vt {:.4f} {:.4f}\n".format(r, c))
            for i in range(num_triangles):
                f.write(
                    "f {:d}/{:d} {:d}/{:d} {:d}/{:d}\n".format(
                        indices[i * 3 + 2] + 1,
                        indices[i * 3 + 2] + 1,
                        indices[i * 3 + 1] + 1,
                        indices[i * 3 + 1] + 1,
                        indices[i * 3] + 1,
                        indices[i * 3] + 1,
                    )
                )

    iter_num = 200
    for i in range(iter_num):
        scene = step(100)
        canvas.scene(scene)
        video_manager.write_frame(window.get_image_buffer_as_numpy())
        saveMesh(i)
        print(f"\rFrame {i+1}/{iter_num} is recorded", end="")

    print()
    print("Exporting .mp4 and .gif videos...")
    video_manager.make_video(mp4=True)
    print(f'MP4 video is saved to {video_manager.get_output_filename(".mp4")}')
    print(f'GIF video is saved to {video_manager.get_output_filename(".gif")}')


def saveJacobi(result_dir):
    from simulator.JacobiSimulator import step

    saveVedioMesh(result_dir, step)


def saveGs(result_dir):
    from simulator.GSSimulator import step

    saveVedioMesh(result_dir, step)


def saveCg(result_dir):
    from simulator.CGSimulator import step

    saveVedioMesh(result_dir, step)


if __name__ == "__main__":
    result_dir = "./results"
    # saveJacobi(os.path.join(result_dir, "JacobiSimulator"))
    # saveGs(os.path.join(result_dir, "GSSimulator"))
    saveCg(os.path.join(result_dir, "CGSimulator"))
