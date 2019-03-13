import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot3D(X,Y,Z):

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z, cmap='plasma')
    axis = ["Ro", "Alpha", "mAP"]
    ax.set_xlabel(axis[0])
    ax.set_ylabel(axis[1])
    ax.set_zlabel(axis[2])
    plt.savefig('grid_search.png', dpi=300)
    plt.show()

def plot2D(X,Y, xlab, ylab, tit):
    plt.plot(X, Y, linewidth=2.0)
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.title(tit)
    # plt.text(60, .025, r'$\mu=100,\ \sigma=15$')
    # plt.axis([, 160, 0, 0.03])
    plt.grid(True)
    plt.show()