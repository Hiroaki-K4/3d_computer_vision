import numpy as np
import math
import matplotlib.pyplot as plt


def plot_base(elev=25, azim=-70):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection="3d")
    ax.view_init(elev=elev, azim=azim)
    ax.set(xlim=(-1, 1), ylim=(-1, 1), zlim=(-1, 1))
    ax.set(xlabel="X", ylabel="Y", zlabel="Z")
    ax.xaxis.pane.fill = ax.yaxis.pane.fill = ax.zaxis.pane.fill = False

    t = np.linspace(0, 2 * np.pi, 128 + 1)
    alpha = 0.7
    # Plot arc
    ax.plot(np.cos(t), np.sin(t), [0] * len(t), linestyle=":", c="red", alpha=alpha)
    ax.plot(np.cos(t), [0] * len(t), np.sin(t), linestyle=":", c="red", alpha=alpha)
    ax.plot([0] * len(t), np.cos(t), np.sin(t), linestyle=":", c="red", alpha=alpha)
    # Plot axis
    ax.plot([-1, 1], [0, 0], [0, 0], linestyle=":", c="red", alpha=alpha)
    ax.plot([0, 0], [-1, 1], [0, 0], linestyle=":", c="red", alpha=alpha)
    ax.plot([0, 0], [0, 0], [-1, 1], linestyle=":", c="red", alpha=alpha)

    return ax


def get_3d_pos(equi_x, equi_y, width, height):
    # Move the center of image
    moved_x = equi_x - width / 2 + 0.5
    moved_y = (-1) * equi_y + height / 2 + 0.5
    # Normalize coordinates
    norm_x = moved_x / (width / 2 - 0.5)
    norm_y = moved_y / (height / 2 - 0.5)
    # Calculate longitude and latitude
    longitude = norm_x * math.pi
    latitude = norm_y * math.pi / 2

    x = math.cos(latitude) * math.sin(longitude)
    y = math.cos(latitude) * math.cos(longitude)
    z = math.sin(latitude)

    return x, y, z


def main():
    width = 5760
    height = 2880
    equi_pos_list = [
        [0, height / 2 - 1],
        [width / 2 - 1, 0],
        [width / 2 - 1, height / 2 - 1],
        [width / 2 - 1, height - 1],
        [width - 1, height / 2 - 1],
    ]
    sphere_x_list = []
    sphere_y_list = []
    sphere_z_list = []
    for equi_pos in equi_pos_list:
        print("Equirectangular: ", equi_pos[0], equi_pos[1])
        x, y, z = get_3d_pos(equi_pos[0], equi_pos[1], width, height)
        print("Sphere: ", x, y, z)
        sphere_x_list.append(x)
        sphere_y_list.append(y)
        sphere_z_list.append(z)

    ax = plot_base()
    ax.scatter(sphere_x_list, sphere_y_list, sphere_z_list, s=40, c="green")
    plt.show()


if __name__ == "__main__":
    main()
