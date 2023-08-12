import os
from PIL import Image

def resize_images_in_directory(directory, width, height):
    try:
        for filename in os.listdir(directory):
            if filename.endswith(('.jpg', '.jpeg', '.png', '.gif')):
                full_path = os.path.join(directory, filename)
                image = Image.open(full_path)
                resized_image = image.resize((width, height), Image.ANTIALIAS)
                resized_image.save(full_path)
                print(f"Image {filename} resized successfully!")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    directory = "path_to_images_directory"
    desired_width = 640
    desired_height = 640

    resize_images_in_directory(directory, desired_width, desired_height)
