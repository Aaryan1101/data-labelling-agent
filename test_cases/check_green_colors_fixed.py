"""
Check Green Colors in Annotated Image - Fixed
"""
import cv2
import numpy as np

print("Checking Green Colors in Annotated Image")
print("=" * 40)

# Load the annotated image
image = cv2.imread("final_agent_annotated.jpg")

# Convert to HSV
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Find all green pixels (broader range)
green_mask = cv2.inRange(hsv, np.array([0, 255, 0]), np.array([179, 255, 255]))
green_pixels = hsv[green_mask == 255]

print(f"Total green pixels found: {len(green_pixels)}")

if len(green_pixels) > 0:
    # Get HSV values of green pixels
    green_hsv = hsv[green_mask]
    
    # Calculate min/max HSV values
    h_vals = green_hsv[:, :, 0].flatten()
    s_vals = green_hsv[:, :, 1].flatten()
    v_vals = green_hsv[:, :, 2].flatten()
    
    h_min = np.min(h_vals)
    h_max = np.max(h_vals)
    s_min = np.min(s_vals)
    s_max = np.max(s_vals)
    v_min = np.min(v_vals)
    v_max = np.max(v_vals)
    
    print(f"Green HSV ranges:")
    print(f"  Hue: {h_min} to {h_max}")
    print(f"  Saturation: {s_min} to {s_max}")
    print(f"  Value: {v_min} to {v_max}")
    
    # Suggested ranges for detection
    print(f"\nSuggested HSV ranges for green detection:")
    print(f"  Hue: {max(0, h_min-10)} to {min(179, h_max+10)}")
    print(f"  Saturation: {max(30, s_min-10)} to 255")
    print(f"  Value: {max(30, v_min-10)} to 255")
    
    # Test with broader range
    broader_mask = cv2.inRange(hsv, np.array([max(0, h_min-10), max(30, s_min-10), max(30, v_min-10)]), 
                                     np.array([min(179, h_max+10), 255, 255]))
    broader_count = np.sum(broader_mask == 255)
    print(f"\nBroader range detection: {broader_count} green pixels")
    
else:
    print("No green pixels found in the image")

print("\n" + "=" * 40)
print("GREEN COLOR ANALYSIS COMPLETE")
