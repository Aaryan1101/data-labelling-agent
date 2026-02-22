"""
Check Green Colors in Annotated Image
"""
import cv2
import numpy as np

print("Checking Green Colors in Annotated Image")
print("=" * 40)

# Load the annotated image
image = cv2.imread("final_agent_annotated.jpg")

# Sample some green pixels from the image
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Find all green pixels
green_mask = cv2.inRange(hsv, np.array([0, 255, 0]), np.array([179, 255, 255]))
green_pixels = image[np.all(green_mask == 255, axis=2)]

print(f"Total green pixels found: {len(green_pixels)}")

if len(green_pixels) > 0:
    # Get HSV values of green pixels
    hsv_values = hsv[np.all(green_mask == 255, axis=2)]
    
    # Calculate min/max HSV values
    h_min = np.min(hsv_values[:, :, 0])
    h_max = np.max(hsv_values[:, :, 0])
    s_min = np.min(hsv_values[:, :, 1])
    s_max = np.max(hsv_values[:, :, 1])
    v_min = np.min(hsv_values[:, :, 2])
    v_max = np.max(hsv_values[:, :, 2])
    
    print(f"Green HSV ranges:")
    print(f"  Hue: {h_min} to {h_max}")
    print(f"  Saturation: {s_min} to {s_max}")
    print(f"  Value: {v_min} to {v_max}")
    
    # Suggested ranges
    print(f"\nSuggested HSV ranges for green detection:")
    print(f"  Hue: {h_min-10} to {h_max+10}")
    print(f"  Saturation: {max(30, s_min-10)} to 255")
    print(f"  Value: {max(30, v_min-10)} to 255")
else:
    print("No green pixels found in the image")

print("\n" + "=" * 40)
print("GREEN COLOR ANALYSIS COMPLETE")
