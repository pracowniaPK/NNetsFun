import matplotlib.pyplot as plt


def show_grid(imgs, x, y, save_path=None, show=True):
    """Draws grid of images from array of matrices

    Parameters:
        imgs: (n,x,y) array of x*y matrices to draw
        x: number of columns of grid
        y: number of rows of grid
        save_path: if non-empty saves grid in given location
        show: if True shows the drawn grid
    """
    if imgs.shape[0] != x*y:
        raise ValueError('grid size ({} x {}) doesnt match input size {}'
            .format(x, y, imgs.shape[0]))

    for i in range(x):
        for j in range(y):
            plt.subplot(y, x, y*i+j+1)
            plt.imshow(imgs[y*i+j], cmap=plt.get_cmap('gist_gray'))
            plt.axis('off')

    if save_path:
        plt.savefig(save_path)
    if show:
        plt.show()
