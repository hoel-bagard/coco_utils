import os

import cv2
import numpy as np
import numpy.typing as npt


def show_img(img: npt.NDArray[np.uint8] | cv2.Mat, window_name: str = "Image", is_bgr: bool = True) -> None:
    """Display the given image.

    If a display (monitor) is detected, then display the image on the screen until the user presses the "q" key.
    Otherwise try to display the image to the terminal.

    Args:
        img: The image that is to be displayed.
        window_name: The name of the window in which the image will be displayed.
        is_bgr: Should be True if the image format is BGR, False otherwise.
    """
    if "DISPLAY" in os.environ:  # TODO: check if that works on Windows too.
        while True:
            # Make the image full screen if it's above a given size (assume the screen isn't too small^^)
            if any(img.shape[:2] > np.asarray([1080, 1440])):
                cv2.namedWindow(window_name, cv2.WND_PROP_FULLSCREEN)
                cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

            if not is_bgr and img.shape[2] == 3:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            cv2.imshow(window_name, img)
            key = cv2.waitKey(10)
            if key == ord("q"):
                cv2.destroyAllWindows()
                break
    else:
        try:
            from PIL import Image
            from term_image.image import AutoImage  # type: ignore

            if is_bgr:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            AutoImage(Image.fromarray(img)).draw()
        except ModuleNotFoundError:
            if "warning_printed" not in show_img.__dict__:
                show_img.warning_printed = True
                print("Consider installing the term_image and Pillow packages to display images in the terminal.")
                print("You can do that using:\n\tpip install term_image Pillow")
