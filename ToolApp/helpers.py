import numpy as np

# combines snapshots together in one large image
def concatImages(images):
    # Concatonate an array of images into one large one
    max_width = 0  # find the max width of all the images
    total_height = 0  # the total height of the images (vertical stacking)

    for i in images:
        # open all images and find their sizes

        if i.shape[1] > max_width:
            max_width = i.shape[1]
        total_height += i.shape[0]

    final_image = np.zeros((total_height, max_width), dtype=np.uint8)

    # for imag in self.pics[1:]:
    # imgh = cv2.hconcat([imgh, imag])

    current_y = 0  # keep track of where your current image was last placed in the y coordinate
    for image in images:
        # add an image to the final array and increment the y coordinate
        final_image[current_y:image.shape[0] + current_y, :image.shape[1]] = image
        current_y += image.shape[0]

    return final_image