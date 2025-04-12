import numpy as np


def compute_sobel_gradients_two_loops(
    image: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Computes Sobel gradients along the x and y axes.

    Args:
        image (np.ndarray): The input image.

    Returns:
        tuple[np.ndarray, np.ndarray]: A tuple containing the gradients along
                                       the x-axis and y-axis, respectively.

    """
    # Get image dimensions
    height, width = image.shape

    # Initialize output gradients
    gradient_x = np.zeros_like(image, dtype=np.float64)
    gradient_y = np.zeros_like(image, dtype=np.float64)

    # Pad the image with zeros to handle borders
    padded_image = np.pad(image, ((1, 1), (1, 1)), mode="constant", constant_values=0)
    #  __________end of block__________

    # Define the Sobel kernels for X and Y gradients
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

    # Apply Sobel filter for X and Y gradients using convolution
    for i in range(1, height + 1):
        for j in range(1, width + 1):
            image_part_x = padded_image[
                i - sobel_x.shape[0] // 2 : i + sobel_x.shape[0] // 2 + 1,
                j - sobel_x.shape[1] // 2 : j + sobel_x.shape[1] // 2 + 1,
            ]
            image_part_y = padded_image[
                i - sobel_y.shape[0] // 2 : i + sobel_y.shape[0] // 2 + 1,
                j - sobel_y.shape[1] // 2 : j + sobel_y.shape[1] // 2 + 1,
            ]
            gradient_x[i - 1, j - 1] = np.sum(sobel_x * image_part_x)
            gradient_y[i - 1, j - 1] = np.sum(sobel_y * image_part_y)
    return gradient_x, gradient_y


def compute_gradient_magnitude(sobel_x: np.ndarray, sobel_y: np.ndarray) -> np.ndarray:
    """
    Compute the magnitude of the gradient given the x and y gradients.

    Inputs:
        sobel_x: numpy array of the x gradient.
        sobel_y: numpy array of the y gradient.

    Returns:
        magnitude: numpy array of the same shape as the input [0] with the magnitude of the gradient.
    """
    return np.sqrt(sobel_x**2 + sobel_y**2)


def compute_gradient_direction(sobel_x: np.ndarray, sobel_y: np.ndarray) -> np.ndarray:
    """
    Compute the direction of the gradient given the x and y gradients. Angle must be in degrees in the range (-180; 180].
    Use arctan2 function to compute the angle.

    Inputs:
        sobel_x: numpy array of the x gradient.
        sobel_y: numpy array of the y gradient.

    Returns:
        gradient_direction: numpy array of the same shape as the input [0] with the direction of the gradient.
    """
    return np.degrees(np.arctan2(sobel_y, sobel_x))


def compute_hog(
    image: np.ndarray,
    pixels_per_cell: tuple[int, int] = (7, 7),
    bins: int = 9,
) -> np.ndarray:
    """
    Computes HoG histogram.

    Args:
        image (np.ndarray): The input image.
        pixels_per_cell (tuple[int, int]): The size of pixels per cell.
        bins (int, optional): The bin count of histogram. Defaults to 9.

    Returns:
        np.ndarray: The estimated histogram.
    """
    # 1. Convert the image to grayscale if it's not already (assuming the image is in RGB or BGR)
    if len(image.shape) == 3:
        image = np.mean(image, axis=2)  # Simple averaging to convert to grayscale

    # 2. Compute gradients with Sobel filter
    gradient_x, gradient_y = compute_sobel_gradients_two_loops(image)

    # 3. Compute gradient magnitude and direction
    magnitude = compute_gradient_magnitude(gradient_x, gradient_y)
    direction = compute_gradient_direction(gradient_x, gradient_y)

    # 4. Create histograms of gradient directions for each cell
    cell_height, cell_width = pixels_per_cell
    n_cells_x = image.shape[1] // cell_width
    n_cells_y = image.shape[0] // cell_height

    histograms = np.zeros((n_cells_y, n_cells_x, bins))
    bin_edges = np.linspace(-180, 180, bins + 1)
    for i in range(n_cells_y):
        for j in range(n_cells_x):
            magn_flat = magnitude[
                i * cell_height : (i + 1) * cell_height,
                j * cell_width : (j + 1) * cell_width,
            ].ravel()
            dir_flat = direction[
                i * cell_height : (i + 1) * cell_height,
                j * cell_width : (j + 1) * cell_width,
            ].ravel()
            hist_bins, _ = np.histogram(dir_flat, bins=bin_edges, weights=magn_flat)

            hist_sum = np.sum(hist_bins)
            histograms[i, j, :] = hist_bins / hist_sum if hist_sum != 0 else hist_bins
    return histograms
