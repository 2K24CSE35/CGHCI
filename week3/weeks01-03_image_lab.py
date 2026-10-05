import cv2
import numpy as np
import matplotlib.pyplot as plt


def inspect_image(image_path: str) -> dict:
    """Load the image and return its measured image-data properties."""
    image = cv2.imread(image_path)

    if image is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")

    height, width, channels = image.shape

    pixel_count = width * height
    estimated_bytes = width * height * channels

    return {
        "width": width,
        "height": height,
        "channels": channels,
        "shape": list(image.shape),
        "pixel_count": pixel_count,
        "estimated_bytes": estimated_bytes,
        "color_order": "BGR",
    }


def create_pixel_views(image_path: str, output_dir: str) -> dict:
    """Create the labeled channel, grayscale, and downsampled views."""
    image = cv2.imread(image_path)

    if image is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")


    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

  
    red = rgb[:, :, 0]
    green = rgb[:, :, 1]
    blue = rgb[:, :, 2]


    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


    height, width = gray.shape
    new_width = width // 2
    new_height = height // 2

    downsampled = cv2.resize(gray, (new_width, new_height))

    fig, axes = plt.subplots(2, 3, figsize=(12, 8))

    axes[0, 0].imshow(red, cmap="gray")
    axes[0, 0].set_title("Red Channel")

    axes[0, 1].imshow(green, cmap="gray")
    axes[0, 1].set_title("Green Channel")

    axes[0, 2].imshow(blue, cmap="gray")
    axes[0, 2].set_title("Blue Channel")

    axes[1, 0].imshow(gray, cmap="gray")
    axes[1, 0].set_title("Grayscale")

    axes[1, 1].imshow(downsampled, cmap="gray")
    axes[1, 1].set_title("Half Resolution")

    axes[1, 2].axis("off")

    for ax in axes.flat:
        ax.axis("off")

    plt.tight_layout()

    output_path = f"{output_dir}/pixel_views.png"
    plt.savefig(output_path)
    plt.close()

    return {
        "original_size": [width, height],
        "downsampled_size": [new_width, new_height],
        "output_path": output_path,
    }


def create_adjustments(
    image_path: str,
    output_dir: str,
    brightness_delta: int = 40,
    contrast_factor: float = 1.5,
    threshold: int = 127,
) -> dict:
    """Create labeled brightness, contrast, and threshold results."""

    if not 0 <= threshold <= 255:
        raise ValueError("threshold must be between 0 and 255")

    image = cv2.imread(image_path)

    if image is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")

 
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

   
    brighter = cv2.add(gray, brightness_delta)

   
    higher_contrast = np.clip(
        gray.astype(np.float32) * contrast_factor,
        0,
        255,
    ).astype(np.uint8)

    # Threshold
    _, black_white = cv2.threshold(
        gray,
        threshold,
        255,
        cv2.THRESH_BINARY,
    )

   
    fig, axes = plt.subplots(1, 4, figsize=(16, 5))

    axes[0].imshow(gray, cmap="gray")
    axes[0].set_title("Original Grayscale")

    axes[1].imshow(brighter, cmap="gray")
    axes[1].set_title("Brighter (+40)")

    axes[2].imshow(higher_contrast, cmap="gray")
    axes[2].set_title("Higher Contrast (1.5x)")

    axes[3].imshow(black_white, cmap="gray")
    axes[3].set_title("Threshold (127)")

    for ax in axes:
        ax.axis("off")

    plt.tight_layout()

    output_path = f"{output_dir}/adjustments.png"
    plt.savefig(output_path)
    plt.close()

    return {
        "brightness_delta": brightness_delta,
        "contrast_factor": contrast_factor,
        "threshold": threshold,
        "output_path": output_path,
    }


def create_blur_and_edges(
    image_path: str,
    output_dir: str,
    kernel_size: int = 5,
) -> dict:
    """Create labeled grayscale, mean-blur, and Sobel-edge results."""

    if kernel_size <= 0 or kernel_size % 2 == 0:
        raise ValueError("kernel_size must be a positive odd integer")

    image = cv2.imread(image_path)

    if image is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Mean blur
    blurred = cv2.blur(gray, (kernel_size, kernel_size))

    # Sobel edges from original grayscale
    sobel_x_original = cv2.Sobel(
        gray,
        cv2.CV_64F,
        1,
        0,
        ksize=3,
    )

    sobel_y_original = cv2.Sobel(
        gray,
        cv2.CV_64F,
        0,
        1,
        ksize=3,
    )

    edges_original = cv2.magnitude(
        sobel_x_original,
        sobel_y_original,
    )

    
    sobel_x_blurred = cv2.Sobel(
        blurred,
        cv2.CV_64F,
        1,
        0,
        ksize=3,
    )

    sobel_y_blurred = cv2.Sobel(
        blurred,
        cv2.CV_64F,
        0,
        1,
        ksize=3,
    )

    edges_blurred = cv2.magnitude(
        sobel_x_blurred,
        sobel_y_blurred,
    )

    # Create labeled montage
    fig, axes = plt.subplots(1, 4, figsize=(16, 5))

    axes[0].imshow(gray, cmap="gray")
    axes[0].set_title("Grayscale")

    axes[1].imshow(blurred, cmap="gray")
    axes[1].set_title("Mean Blur (5x5)")

    axes[2].imshow(edges_original, cmap="gray")
    axes[2].set_title("Sobel - Original")

    axes[3].imshow(edges_blurred, cmap="gray")
    axes[3].set_title("Sobel - Blurred")

    for ax in axes:
        ax.axis("off")

    plt.tight_layout()

    output_path = f"{output_dir}/blur_and_edges.png"
    plt.savefig(output_path)
    plt.close()

    return {
        "kernel_size": kernel_size,
        "output_path": output_path,
    }


def run_lab(image_path: str, output_dir: str) -> dict:
    """Run Tasks 1–4 and return their results together."""

    
    import os
    os.makedirs(output_dir, exist_ok=True)

    task1 = inspect_image(image_path)

    task2 = create_pixel_views(
        image_path,
        output_dir,
    )

    task3 = create_adjustments(
        image_path,
        output_dir,
    )

    task4 = create_blur_and_edges(
        image_path,
        output_dir,
    )

    return {
        "task1": task1,
        "task2": task2,
        "task3": task3,
        "task4": task4,
    }


def main() -> None:
    """Run the lab using the required repository paths."""

    results = run_lab(
        "images/original.jpg",
        "outputs",
    )

    print("Image Lab completed successfully.")
    print()
    print("Task 1:")
    print(results["task1"])

    print()
    print("Task 2:")
    print(results["task2"])

    print()
    print("Task 3:")
    print(results["task3"])

    print()
    print("Task 4:")
    print(results["task4"])


if __name__ == "__main__":
    main()