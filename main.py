import numpy as np
from PIL import Image
from scipy import ndimage
from random import randint
from matplotlib import pyplot as plt
from skimage import transform
from skimage import img_as_ubyte
import time
from matplotlib.widgets import RectangleSelector
from imageio import imwrite

# Options are: "decrease_width", "decrease_height", "decrease_both", "increase_width",
# "increase_height", "increase_both", "amplify_content" or "remove_object"
resize_option = 'increase_height'
# Number of pixels to decrease/increase the image
pixels_number = 0
# Whether to plot the minimum path or not
plot_path = False
# The color to display the path if displayed
path_color = np.asarray([255, 0, 0])
# Options are: "random", "greedy" or "dynamic"
select_path_mode = 'greedy'
# How many times to amplify the content if applied
amplify_scale = 1


def rgb2gray(rgb):
    return np.dot(rgb, [0.299, 0.587, 0.114])


def get_image_energy(image: np.ndarray):
    gray_image = rgb2gray(image)
    x_energy = ndimage.sobel(gray_image)
    y_energy = ndimage.sobel(gray_image, axis=0)

    return np.abs(x_energy) + np.abs(y_energy)


# Get the vertical path to remove from image.
# selection: For object removal, specify the user selection to be removed
def get_vertical_seam(image: np.ndarray, select_path_mode: str = 'dynamic', selection: np.ndarray = None):
    energy = get_image_energy(image)
    # Update the image energy to assure that selection pixels are selected
    if selection is not None:
        neutral_energy = image.shape[0] * -energy.max() - 1
        x_start, x_stop = selection[0, 0], selection[1, 0]
        y_start, y_stop = selection[0, 1], selection[1, 1]
        energy[y_start:y_stop, x_start:x_stop] = neutral_energy

    path = np.empty(image.shape[0], dtype=np.uint)
    if select_path_mode == 'random':
        path[0] = randint(0, energy.shape[1])
        for line in range(1, energy.shape[0]):
            last_col = path[line - 1]
            if last_col == 0:
                col = randint(0, 1)
            elif last_col == energy.shape[1] - 1:
                col = randint(energy.shape[1] - 2, energy.shape[1] - 1)
            else:
                col = randint(last_col - 1, last_col + 1)
            path[line] = col
    elif select_path_mode == 'greedy':
        path[0] = energy[0].argmin()
        for line in range(1, energy.shape[0]):
            # Set the next column the same as last_column so by default it would
            # go down the line for the best path
            last_col = col = path[line - 1]
            # See if there is any other better option to choose the path
            if energy[line, max(0, last_col - 1)] < energy[line, col]:
                col = last_col - 1
            if energy[line, min(energy.shape[1] - 1, last_col + 1)] < energy[line, col]:
                col = last_col + 1
            path[line] = col
    elif select_path_mode == 'dynamic':
        min_path = np.empty_like(energy)
        min_path[0] = energy[0]
        for line in range(1, energy.shape[0]):
            for col in range(energy.shape[1]):
                min_path[line, col] = energy[line, col] + min(min_path[line - 1, max(0, col - 1)],
                                                              min_path[line - 1, col],
                                                              min_path[line - 1, min(col + 1, energy.shape[1] - 1)])
        # Go back to get the minimum path
        path[-1] = min_path[-1].argmin()
        for i in range(len(path) - 2, -1, -1):
            last_col = col = path[i + 1]
            if min_path[i, max(0, last_col - 1)] < min_path[i, col]:
                col = last_col - 1
            if min_path[i, min(last_col + 1, min_path.shape[1] - 1)] < min_path[i, col]:
                col = last_col + 1
            path[i] = col

    return path


def remove_vertical_seam(image: np.ndarray, path: np.ndarray):
    new_image = np.array(image[:, :image.shape[1] - 1])
    # Shift the pixels one position to the left on each row
    for i in range(len(path)):
        new_image[i, path[i]:] = image[i, path[i] + 1:]

    return new_image


# noinspection PyUnboundLocalVariable
def decrease_image_width(image: np.ndarray, pixels_number: int, select_path_mode: str = 'dynamic',
                         selection: np.ndarray = None):
    if pixels_number > image.shape[1]:
        print('You can not remove more pixels than the image has, dumbass.')
        return image

    # Sort the start and end positions of the selection
    if selection is not None:
        selection.sort(axis=0)
    # Update the second subplot to display the chosen paths
    if plot_path:
        ax = plt.gcf().axes[1]

    for i in range(pixels_number):
        print("Removing seam %d out of a total of %d seams..." % (i + 1, pixels_number))
        path = get_vertical_seam(image, select_path_mode, selection)

        if plot_path:
            ax.cla()
            ax.imshow(image)
            ax.plot(path, range(image.shape[0]), color=path_color / 255)
            # Pause the plot for a shot time to have enough time to render the image
            plt.pause(.1)

        image = remove_vertical_seam(image, path)
        # One pixel has been removed from the selection, so the end point
        # decreases by one pixel
        if selection is not None:
            selection[1, 0] -= 1

    return image


def increase_width(image: np.ndarray, pixels_number: int, select_path_mode: str = 'dynamic'):
    if pixels_number > image.shape[1]:
        print('You can not more than double the image size.')
        return image

    shape = image.shape
    resized_image = np.empty((image.shape[0], image.shape[1] + pixels_number, image.shape[2]), dtype=np.int16)
    resized_image[:, :image.shape[1]] = image
    path_history = np.empty((pixels_number, image.shape[0]))

    for i in range(pixels_number):
        print("Inserting seam %d out of a total of %d seams..." % (i + 1, pixels_number))
        path = get_vertical_seam(image, select_path_mode)
        # Remove the path from the image so it won't be chosen multiple times
        image = remove_vertical_seam(image, path)
        # Remind which path has been removed
        path_history[i] = path
        # Get the corresponding path in the resized image by adding the number
        # of removed pixels in front of it before
        resized_image_path = path + (path_history[:i] <= path).sum(axis=0)
        left_path = np.maximum(resized_image_path - 1, 0)
        right_path = np.minimum(resized_image_path + 1, shape[1] + i)
        lines = range(shape[0])
        # Compute the rows of pixels to be inserted instead of the removed
        # path by adding the means between the left and right paths and the
        # removed path
        first_replace = (resized_image[lines, left_path] + resized_image[lines, resized_image_path]) // 2
        second_replace = (resized_image[lines, right_path] + resized_image[lines, resized_image_path]) // 2
        # Shift the pixels to the right by one pixel
        for j in range(len(right_path)):
            resized_image[j, right_path[j] + 1:shape[1] + i + 1] = resized_image[j, right_path[j]:shape[1] + i]
        # Place the replacements inside the resized image
        resized_image[lines, resized_image_path] = first_replace
        resized_image[lines, resized_image_path + 1] = second_replace

    return resized_image.astype(np.uint8)


def amplify_content(image: np.ndarray, amplify_scale: float):
    if amplify_scale < 1:
        print('The amplify scale must be greater or equal to 1.')
        return

    print('Scaling the image %.2f times...' % amplify_scale)
    resized_image = img_as_ubyte(transform.rescale(image, amplify_scale, multichannel=True, mode='symmetric',
                                                   anti_aliasing=False))
    amplified_image = resize_image(resized_image, 'decrease_width', resized_image.shape[1] - image.shape[1],
                                   select_path_mode=select_path_mode)
    amplified_image = resize_image(amplified_image, 'decrease_height', resized_image.shape[0] - image.shape[0],
                                   select_path_mode=select_path_mode)
    return amplified_image


# Event handler for user selection
def on_select(e_click, e_release):
    # Remove the selection tool
    selector.RS.set_active(False)
    selector.RS = None
    selection = np.asarray([[e_click.xdata, e_click.ydata], [e_release.xdata, e_release.ydata]], dtype=np.int)
    # Remove the object from the image
    image = decrease_image_width(selector.image, abs(selection[0, 0] - selection[1, 0]),
                                 select_path_mode=select_path_mode, selection=selection)
    selector.image = image
    # Announce that the image is ready
    selector.ready = True


def selector():
    pass


# The main function of the program, which decides how to resize the image
def resize_image(image: np.ndarray, resize_option: str, pixels_number: int = None, amplify_scale: float = None,
                 select_path_mode: str = 'dynamic'):
    if resize_option == 'decrease_width':
        print('Decreasing the image width by %d pixels...' % pixels_number)
        image = decrease_image_width(image, pixels_number, select_path_mode)
        print('Image width has been decreased by %d pixels.' % pixels_number)
    elif resize_option == 'decrease_height':
        print('Decreasing the image height by %d pixels...' % pixels_number)
        rotated_image = np.rot90(image, 3)
        rotated_image = decrease_image_width(rotated_image, pixels_number, select_path_mode)
        image = np.rot90(rotated_image)
        print('Image height has been decreased by %d pixels.' % pixels_number)
    elif resize_option == 'decrease_both':
        image = resize_image(image, 'decrease_width', pixels_number, select_path_mode=select_path_mode)
        image = resize_image(image, 'decrease_height', pixels_number, select_path_mode=select_path_mode)
    elif resize_option == 'increase_width':
        print('Increasing the image width by %d pixels...' % pixels_number)
        image = increase_width(image, pixels_number, select_path_mode)
        print('Image width has been increased by %d pixels.' % pixels_number)
    elif resize_option == 'increase_height':
        print('Increasing the image height by %d pixels...' % pixels_number)
        rotated_image = np.rot90(image, 3)
        rotated_image = increase_width(rotated_image, pixels_number, select_path_mode)
        image = np.rot90(rotated_image)
        print('Image height has been increased by %d pixels.' % pixels_number)
    elif resize_option == 'increase_both':
        image = resize_image(image, 'increase_width', pixels_number, select_path_mode=select_path_mode)
        image = resize_image(image, 'increase_height', pixels_number, select_path_mode=select_path_mode)
    elif resize_option == 'amplify_content':
        image = amplify_content(image, amplify_scale)
        print('Image content has been amplified %.2f times.' % amplify_scale)
    elif resize_option == 'remove_object':
        # Apply the widget on the initial image
        ax = plt.gcf().axes[0]
        selector.image = image
        selector.ready = False
        selector.RS = RectangleSelector(ax, on_select, drawtype='box')
        # Loop while the event is being completed async
        while not selector.ready:
            plt.pause(.1)
        image = selector.image
    elif resize_option != 'none':
        print('No such option defined.')
    return image


def handle_close(_):
    global exit_program
    exit_program = True


image = np.asarray(Image.open('data/square.jpg'), dtype=np.uint8)

print('Showing the initial image...')
fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.set_title('Initial image')
ax1.imshow(image)
plt.pause(.1)

start = time.time()
print('Resizing the image...')
image = resize_image(image, resize_option, pixels_number, amplify_scale, select_path_mode=select_path_mode)
print('Resizing operation finished in %d seconds.' % (time.time() - start))

print('Showing the resulted image...')
ax2.cla()
ax2.set_title('Resulting image')
ax2.imshow(image)

imwrite('square.jpeg', image)

# Create a loop to prevent the ending process from closing the figure
# until the user closed it
exit_program = False
fig.canvas.mpl_connect('close_event', handle_close)
plt.pause(.1)
while not exit_program:
    fig.canvas.get_tk_widget().update()
