import numpy as np


def conv(image, kernel):
    """ An implementation of convolution filter.

    This function uses element-wise multiplication and np.sum()
    to efficiently compute weighted sum of neighborhood at each
    pixel.

    Args:
        image: numpy array of shape (Hi, Wi)
        kernel: numpy array of shape (Hk, Wk)

    Returns:
        out: numpy array of shape (Hi, Wi)
    """
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    out = np.zeros((Hi, Wi))

    # For this assignment, we will use edge values to pad the images.
    # Zero padding will make derivatives at the image boundary very big,
    # whereas we want to ignore the edges at the boundary.
    pad_width0 = Hk // 2
    pad_width1 = Wk // 2
    pad_width = ((pad_width0,pad_width0),(pad_width1,pad_width1))
    padded = np.pad(image, pad_width, mode='edge')


    ### YOUR CODE HERE
    for w in range(Hi):
        for h in range(Wi):
            window = padded[w:w+pad_width0*2 + 1, h: h+pad_width1*2 + 1]
            out[w, h] = np.sum(window * kernel)
    ### END YOUR CODE

    return out


def gaussian_kernel(size, sigma):
    """ Implementation of Gaussian Kernel.

    This function follows the gaussian filter_values formula,
    and creates a filter_values matrix.

    Hints:
    - Use np.pi and np.exp to compute pi and exp

    Args:
        size: int of the size of output matrix
        sigma: float of sigma to calculate filter_values

    Returns:
        filter_values: numpy array of shape (size, size)
    """

    filter_values = np.zeros((size, size))
    delta = (size-1) / 2

    ### YOUR CODE HERE
    for i in range(size):
        for j in range(size):
            filter_values[i, j] = 1 / (2*np.pi*sigma) * np.exp(-((i-delta)**2+(j-delta)**2)/(2*sigma**2))
    ### END YOUR CODE

    return filter_values


def partial_x(img):
    """ Computes partial x-derivative of input img.

    Hints:
        - You may use the conv function in defined in this file.

    Args:
        img: numpy array of shape (H, W)
    Returns:
        out: x-derivative image
    """

    out = None

    ### YOUR CODE HERE
    Dx = np.array(
            [[ 0, 0, 0],
            [ -0.5, 0, 0.5],
            [ 0, 0, 0]]
        , np.float32)

    out = conv(img, Dx)
    ### END YOUR CODE

    return out


def partial_y(img):
    """ Computes partial y-derivative of input img.

    Hints:
        - You may use the conv function in defined in this file.

    Args:
        img: numpy array of shape (H, W)
    Returns:
        out: y-derivative image
    """

    out = None

    ### YOUR CODE HERE
    Dy = np.array(
        [[ 0, -0.5, 0],
        [ 0, 0, 0],
        [ 0, 0.5, 0]]
    , np.float32)

    out = conv(img, Dy)
    ### END YOUR CODE

    return out


def gradient(img):
    """ Returns gradient magnitude and direction of input img.

    Args:
        img: Grayscale image. Numpy array of shape (H, W)

    Returns:
        G: Magnitude of gradient at each pixel in img.
            Numpy array of shape (H, W)
        theta: Direction(in degrees, 0 <= theta < 360) of gradient
            at each pixel in img. Numpy array of shape (H, W)
    """
    G = np.zeros(img.shape)
    theta = np.zeros(img.shape)

    ### YOUR CODE HERE
    gx = partial_x(img)
    gy = partial_y(img)
    G = np.sqrt(gx**2 + gy**2)
    theta = np.degrees(np.arctan2(gy, gx)) + 180
    ### END YOUR CODE

    return G, theta


def non_maximum_suppression(G, theta):
    """ Performs non-maximum suppression

    This function performs non-maximum suppression along the direction
    of gradient (theta) on the gradient magnitude image (G).

    Args:
        G: gradient magnitude image with shape of (H, W)
        theta: direction of gradients with shape of (H, W)

    Returns:
        out: non-maxima suppressed image
    """
    H, W = G.shape
    out = np.zeros((H, W))

    # Round the gradient direction to the nearest 45 degrees
    theta = np.floor((theta + 22.5) / 45) * 45

    ### BEGIN YOUR CODE

    for i in range(H):
        for j in range(W):
            q = 255
            r = 255

            #angle 0
            if (theta[i,j] % 180 == 0):
                q = G[i, j+1] if j+1<W else 0
                r = G[i, j-1] if j-1>=0 else 0
            #angle 45
            elif (theta[i,j] % 180 == 45):
                q = G[i+1, j-1] if (i+1<H and j-1>=0) else 0
                r = G[i-1, j+1] if (j+1<W and i-1>=0) else 0
            #angle 90
            elif (theta[i,j] % 180 == 90):
                q = G[i+1, j] if i+1<H else 0
                r = G[i-1, j] if i-1>=0 else 0
            #angle 135
            elif (theta[i,j] % 180 == 135):
                q = G[i-1, j-1] if (i-1>=0 and j-1>=0) else 0
                r = G[i+1, j+1] if (j+1<W and i+1<H) else 0

            if (G[i,j] >= q) and (G[i,j] >= r):
                out[i,j] = G[i,j]

    ### END YOUR CODE

    return out


def double_thresholding(img, high, low):
    """
    Args:
        img: numpy array of shape (H, W) representing NMS edge response
        high: high threshold(float) for strong edges
        low: low threshold(float) for weak edges

    Returns:
        strong_edges: Boolean array representing strong edges.
            Strong edeges are the pixels with the values above
            the higher threshold.
        weak_edges: Boolean array representing weak edges.
            Weak edges are the pixels with the values below the
            higher threshould and above the lower threshold.
    """

    strong_edges = np.zeros(img.shape)
    weak_edges = np.zeros(img.shape)

    ### YOUR CODE HERE
    strong_i, strong_j = np.where(img >= high)
    zeros_i, zeros_j = np.where(img < low)
    weak_i, weak_j = np.where((img <= high) & (img >= low))

    strong_edges[strong_i, strong_j] = 1
    weak_edges[weak_i, weak_j] = 1
    ### END YOUR CODE

    return strong_edges.astype(bool), weak_edges.astype(bool)


def get_neighbors(y, x, H, W):
    """ Return indices of valid neighbors of (y, x)

    Return indices of all the valid neighbors of (y, x) in an array of
    shape (H, W). An index (i, j) of a valid neighbor should satisfy
    the following:
        1. i >= 0 and i < H
        2. j >= 0 and j < W
        3. (i, j) != (y, x)

    Args:
        y, x: location of the pixel
        H, W: size of the image
    Returns:
        neighbors: list of indices of neighboring pixels [(i, j)]
    """
    neighbors = []

    ### YOUR CODE HERE
    for i in (y-1, y, y+1):
        for j in (x-1, x, x+1):
            if i >= 0 and i < H and j >= 0 and j < W:
                if ((i, j) != (y, x)):
                    neighbors.append((i, j))
    ### END YOUR CODE

    return neighbors


def link_edges(strong_edges, weak_edges):
    """ Find weak edges connected to strong edges and link them.

    Iterate over each pixel in strong_edges and perform breadth first
    search across the connected pixels in weak_edges to link them.
    Here we consider a pixel (a, b) is connected to a pixel (c, d)
    if (a, b) is one of the eight neighboring pixels of (c, d).

    Args:
        strong_edges: binary image of shape (H, W)
        weak_edges: binary image of shape (H, W)
    Returns:
        edges: numpy array of shape(H, W)
    """

    H, W = strong_edges.shape
    indices = np.stack(np.nonzero(strong_edges)).T
    edges = np.zeros((H, W))
    ### YOUR CODE HERE
    for strongindex in indices:
        edges[strongindex[0]][strongindex[1]] = 1
        for n_index in get_neighbors(strongindex[1], strongindex[0], H, W):
            if (weak_edges[n_index[0]][n_index[1]]):
                edges[n_index[0]][n_index[1]] = 1
    ### END YOUR CODE

    return edges


def canny(img, kernel_size=5, sigma=1.4, high=20, low=15):
    """ Implement canny edge detector by calling functions above.

    Args:
        img: binary image of shape (H, W)
        kernel_size: int of size for kernel matrix
        sigma: float for calculating kernel
        high: high threshold for strong edges
        low: low threashold for weak edges
    Returns:
        edge: numpy array of shape(H, W)
    """

    ### YOUR CODE HERE

    kernel = gaussian_kernel(kernel_size, sigma)
    blurred_img = conv(img, kernel)
    G, theta = gradient(blurred_img)
    nms = non_maximum_suppression(G, theta)
    strong_edges, weak_edges = double_thresholding(nms, high, low)
    edge = link_edges(strong_edges, weak_edges)

    ### END YOUR CODE

    return edge
