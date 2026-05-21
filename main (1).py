import cv2
import numpy as np
import matplotlib.pyplot as plt

# =========================
# Load Image
# =========================
image_path = "images.jpeg"

image = cv2.imread(image_path)

if image is None:
    print("Image not found")
    exit()

# Resize image
image = cv2.resize(image, (800, 600))

# Copy for output
output = image.copy()

# =========================
# Preprocessing
# =========================

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Blur
blur = cv2.GaussianBlur(gray, (7, 7), 0)

# Adaptive threshold
thresh = cv2.adaptiveThreshold(
    blur,
    255,
    cv2.ADAPTIVE_THRESH_MEAN_C,
    cv2.THRESH_BINARY_INV,
    21,
    5
)

# Remove noise
kernel = np.ones((5, 5), np.uint8)

thresh = cv2.morphologyEx(
    thresh,
    cv2.MORPH_OPEN,
    kernel
)

thresh = cv2.morphologyEx(
    thresh,
    cv2.MORPH_CLOSE,
    kernel
)

# Edge detection
edges = cv2.Canny(blur, 50, 150)

# =========================
# Find contours
# =========================
contours, _ = cv2.findContours(
    thresh,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE
)

# =========================
# Detect potholes
# =========================
for contour in contours:

    area = cv2.contourArea(contour)

    # Ignore tiny or huge regions
    if area < 500 or area > 10000:
        continue

    x, y, w, h = cv2.boundingRect(contour)

    # Aspect ratio filtering
    aspect_ratio = w / float(h)

    if aspect_ratio < 0.4 or aspect_ratio > 2.5:
        continue

    # Severity
    if area < 2000:
        severity = "Low"
        color = (0, 255, 0)

    elif area < 5000:
        severity = "Medium"
        color = (0, 255, 255)

    else:
        severity = "High"
        color = (0, 0, 255)

    # Circle pothole
    center_x = x + w // 2
    center_y = y + h // 2

    radius = int((w + h) / 4)

    cv2.circle(
        output,
        (center_x, center_y),
        radius,
        color,
        3
    )

    # Draw contour
    cv2.drawContours(
        output,
        [contour],
        -1,
        color,
        2
    )

    # Label
    cv2.putText(
        output,
        f"Pothole - {severity}",
        (x, y - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        color,
        2
    )

# =========================
# Show Results
# =========================
plt.figure(figsize=(18, 6))

# Original
plt.subplot(1, 3, 1)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title("Original Image")
plt.axis("off")

# Threshold
plt.subplot(1, 3, 2)
plt.imshow(thresh, cmap='gray')
plt.title("Processed Image")
plt.axis("off")

# Output
plt.subplot(1, 3, 3)
plt.imshow(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
plt.title("Pothole Detection")
plt.axis("off")

plt.tight_layout()
plt.show()