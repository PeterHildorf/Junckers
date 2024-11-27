import cv2
import numpy as np
import math
import matplotlib.pyplot as plt
import os

# Sti til mappen for behandlede billeder
processed_path = "./Processed"



def calculate_angle(line1, line2):
    """
    Beregner vinklen mellem to linjer givet i formatet ((x1, y1), (x2, y2)).
    """
    if len(line1) != 4 or len(line2) != 4:
        raise ValueError("Linjer skal have formatet (x1, y1), (x2, y2)")

    x1, y1, x2, y2 = line1
    x3, y3, x4, y4 = line2

    # Beregn retning af hver linje
    vector1 = [x2 - x1, y2 - y1]
    vector2 = [x4 - x3, y4 - y3]

    # Beregn vinklen mellem vektorerne
    dot_product = np.dot(vector1, vector2)
    mag1 = math.sqrt(vector1[0]**2 + vector1[1]**2)
    mag2 = math.sqrt(vector2[0]**2 + vector2[1]**2)

    # Undgå division med nul
    if mag1 == 0 or mag2 == 0:
        raise ValueError("En af linjerne har nul længde, kan ikke beregne vinkel.")

    angle = math.acos(dot_product / (mag1 * mag2)) * (180 / math.pi)
    return angle

def crop_region(image, region="top", crop_ratio=0.2):
    """
    Beskær top eller bund af billedet.
    """
    h, w = image.shape[:2]
    if region == "top":
        return image[:int(h * crop_ratio), :]
    elif region == "bottom":
        return image[int(h * (1 - crop_ratio)):, :]
    else:
        raise ValueError("Region skal være 'top' eller 'bottom'.")

def process_region(region_image):
    """
    Finder linjer i en beskåret region.
    """
    gray = cv2.cvtColor(region_image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50, minLineLength=30, maxLineGap=5)
    return lines

# Liste over billedstier
image_paths = [
    "./billeder/Korrektvinkel.png",
    "./billeder/forkertvinkel.png",
    "./billeder/forkertvinkel2.png"
]

# Tolerance for at vurdere, om vinklen er korrekt
tolerance = 0.132

# Iterer gennem hver billedsti
for image_path in image_paths:
    print(f"Tjekker billede: {image_path}")
    image = cv2.imread(image_path)

    if image is None:
        print(f"Kunne ikke indlæse billedet: {image_path}")
        continue

    # Beskær top-regionen
    top_region = crop_region(image, region="top")

    # Find linjer i top-regionen
    lines = process_region(top_region)

    if lines is not None and len(lines) >= 2:
        # Tag de første to linjer og beregn vinklen
        line1 = lines[0][0]
        line2 = lines[1][0]
        angle = calculate_angle(line1, line2)
        angle = round(angle, 2)

        # Tjek, om vinklen er inden for tolerance
        if abs(angle - 90) <= tolerance:
            print(f"Vinklen i øverste ende af {image_path} er: {angle:.2f} grader (Korrekt)")
            line_color = (0, 255, 0)  # Grøn for korrekt
        else:
            print(f"Vinklen i øverste ende af {image_path} er: {angle:.2f} grader (Fejl)")
            line_color = (0, 0, 255)  # Rød for fejl

        # Tegn linjerne
        top_region_with_lines = top_region.copy()
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(top_region_with_lines, (x1, y1), (x2, y2), line_color, 2)

        # Gem det behandlede billede
        processed_image_path = os.path.join(processed_path, os.path.basename(image_path))
        cv2.imwrite(processed_image_path, top_region_with_lines)
        print(f"Gemte det behandlede billede som: {processed_image_path}")

    else:
        print(f"Ikke nok linjer fundet i øverste ende af {image_path}.")
