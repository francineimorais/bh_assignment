import os
import random
from PIL import Image, ImageOps, ImageEnhance, ImageFilter

# List of transformations
# Vertical Flips
# Rotation: Rotating the image by certain degrees (e.g., 90, 180, or 270 degrees).
# Brightness Adjustment: Changing the brightness of the image to simulate different lighting conditions.
# Contrast Adjustment: Adjusting the contrast to emphasize or reduce the differences between pixel values.
# Color Jittering: Slight variations in color to simulate different lighting conditions.
# Noise Addition: Adding random noise (e.g., Gaussian noise) to the image.
# Blur: Applying blurring filters to simulate motion or defocus.
# Elastic Distortion: Deforming the image with elastic transformations to simulate deformation.
# Channel Swapping: Swapping color channels (e.g., RGB) to create color variations.
# Random exposure adjustment of between -25 and +25 percent
# Random Gaussian blur of between 0 and 10 pixels
# Salt and pepper noise was applied to 5 percent of pixels


def apply_horizontal_flip(image):
    return ImageOps.mirror(image)

def apply_vertical_flip(image):
    return image.transpose(Image.FLIP_TOP_BOTTOM)

def apply_rotation(image, angle):
    return image.rotate(angle)

def apply_brightness(image, factor):
    enhancer = ImageEnhance.Brightness(image)
    return enhancer.enhance(factor)

def apply_contrast(image, factor):
    enhancer = ImageEnhance.Contrast(image)
    return enhancer.enhance(factor)

def apply_color_jitter(image, factor):
    return Image.blend(image, Image.new('RGB', image.size, (int(factor * 255),) * 3), 0.5)

def apply_noise(image):
    noise = Image.new('RGB', image.size, (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)))
    return Image.blend(image, noise, 0.1)

def apply_blur(image, radius):
    return image.filter(ImageFilter.GaussianBlur(radius))

def apply_elastic_deformation(image):
    elastic_deformation = image.transform(image.size, Image.AFFINE, (1, random.uniform(-0.1, 0.1), random.uniform(-0.1, 0.1), 0, 1, 0), Image.BILINEAR)
    return elastic_deformation

def apply_channel_swap(image):
    r, g, b = image.split()
    shuffled_channels = [r, g, b]
    random.shuffle(shuffled_channels)
    return Image.merge('RGB', shuffled_channels)

def apply_exposure_adjustment(image, factor):
    enhancer = ImageEnhance.Brightness(image)
    return enhancer.enhance(factor)

def apply_gaussian_blur(image, radius):
    return image.filter(ImageFilter.GaussianBlur(radius))

def apply_salt_and_pepper(image):
    salt_and_pepper_ratio = 0.05
    num_salt_and_pepper_pixels = int(image.width * image.height * salt_and_pepper_ratio)
    new_image = image.copy()
    for _ in range(num_salt_and_pepper_pixels):
        x = random.randint(0, new_image.width - 1)
        y = random.randint(0, new_image.height - 1)
        if random.random() < 0.5:
            new_image.putpixel((x, y), (0, 0, 0))
        else:
            new_image.putpixel((x, y), (255, 255, 255))
    return new_image

def apply_all_transformations(image):
    transformations = [
        apply_horizontal_flip,
        apply_vertical_flip,
        lambda img: apply_rotation(img, 90),
        lambda img: apply_rotation(img, 180),
        lambda img: apply_rotation(img, 270),
        lambda img: apply_brightness(img, random.uniform(0.75, 1.25)),
        lambda img: apply_contrast(img, random.uniform(0.75, 1.25)),
        lambda img: apply_color_jitter(img, random.uniform(0.9, 1.1)),
        apply_noise,
        lambda img: apply_blur(img, random.randint(0, 10)),
        apply_elastic_deformation,
        apply_channel_swap,
        lambda img: apply_exposure_adjustment(img, random.uniform(0.75, 1.25)),
        lambda img: apply_gaussian_blur(img, random.randint(0, 10)),
        apply_salt_and_pepper
    ]
    augmented_images = []

    for transformation in transformations:
        augmented_image = transformation(image.copy())
        augmented_images.append(augmented_image)

    return augmented_images

def augment_images_in_directory(directory):
    try:
        for filename in os.listdir(directory):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.gif')):
                full_path = os.path.join(directory, filename)
                image = Image.open(full_path)
                augmented_images = apply_all_transformations(image)

                for i, augmented_image in enumerate(augmented_images):
                    transformation_name = transformations[i].__name__.replace('apply_', '').replace('_', ' ').title()
                    new_filename = f"{os.path.splitext(filename)[0]}_{transformation_name}.png"
                    new_full_path = os.path.join(directory, new_filename)
                    augmented_image.save(new_full_path)
                    print(f"{transformation_name} applied to {filename} - Saved as {new_filename}")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    directory = "path_to_images_directory"

    if not os.path.exists(directory):
        print("Directory not found.")
    else:
        transformations = [
            apply_horizontal_flip,
            apply_vertical_flip,
            lambda img: apply_rotation(img, 90),
            lambda img: apply_rotation(img, 180),
            lambda img: apply_rotation(img, 270),
            lambda img: apply_brightness(img, random.uniform(0.75, 1.25)),
            lambda img: apply_contrast(img, random.uniform(0.75, 1.25)),
            lambda img: apply_color_jitter(img, random.uniform(0.9, 1.1)),
            apply_noise,
            lambda img: apply_blur(img, random.randint(0, 10)),
            apply_elastic_deformation,
            apply_channel_swap,
            lambda img: apply_exposure_adjustment(img, random.uniform(0.75, 1.25)),
            lambda img: apply_gaussian_blur(img, random.randint(0, 10)),
            apply_salt_and_pepper
        ]
        augment_images_in_directory(directory)
