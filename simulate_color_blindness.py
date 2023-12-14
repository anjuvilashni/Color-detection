import numpy as np
import cv2
from colorblind import colorblind
import matplotlib.pyplot as plt


def simulate_color(image_path, simulation_type):
    # Load the image
    img = cv2.imread(image_path)
    img = img[..., ::-1]

    # Simulate color blindness
    simulated_img = colorblind.simulate_colorblindness(
        img, colorblind_type=simulation_type
    )

    cv2.imwrite(filename="upload/simulate.jpg", img=simulated_img)
