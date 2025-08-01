import cv2
import numpy as np

input_image_path = "./utils/watermarked_product_img/watermarked_product9.jpg"
image = cv2.imread(input_image_path)
if image is None:
    raise FileNotFoundError(f"Image not found at path: {input_image_path}")

'''
The function below use to apply various image processing attacks on the watermarked image.
For testing the robustness of the watermark, you can apply different attacks:
Namely JPEG compression, rotation, cropping, brightness adjustment, scaling, Gaussian noise, and salt-and-pepper noise.
'''

def jpeg_compression(image, quality=25):
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    _, encimg = cv2.imencode('.jpg', image, encode_param)
    decimg = cv2.imdecode(encimg, 1)
    return decimg

def rotation(image, angle=30):
    height, width = image.shape[:2]
    center = (width // 2, height // 2)
    rot_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, rot_matrix, (width, height))
    return rotated

def cropping(image, crop_ratio=0.8):
    height, width = image.shape[:2]
    new_h, new_w = int(height * crop_ratio), int(width * crop_ratio)
    start_x = (width - new_w) // 2
    start_y = (height - new_h) // 2
    cropped = image[start_y:start_y + new_h, start_x:start_x + new_w]
    return cv2.resize(cropped, (width, height))

def brightness_adjustment(image, factor=1.5):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv = np.array(hsv, dtype=np.float64)
    hsv[..., 2] *= factor
    hsv[..., 2][hsv[..., 2] > 255] = 255
    hsv = np.array(hsv, dtype=np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

def scaling(image, scale_factor):
    height, width = image.shape[:2]
    scaled_img = cv2.resize(image, (int(width * scale_factor), int(height * scale_factor)))
    return scaled_img

def gaussian_noise(image, mean=0, stddev=10):
    noise = np.random.normal(mean, stddev, image.shape).astype(np.uint8)
    noisy_image = cv2.add(image, noise)
    return noisy_image

def salt_pepper_noise(image, density=0.05):
    output = np.copy(image)
    total_pixels = image.shape[0] * image.shape[1]
    num_salt = int(total_pixels * density / 2)
    num_pepper = int(total_pixels * density / 2)

    # Salt
    coords = [np.random.randint(0, i - 1, num_salt) for i in image.shape[:2]]
    output[coords[0], coords[1]] = 255

    # Pepper
    coords = [np.random.randint(0, i - 1, num_pepper) for i in image.shape[:2]]
    output[coords[0], coords[1]] = 0
    return output

attacked = scaling(image, 0.5)  # or gaussian_noise(image, 0, 25), etc.
cv2.imwrite("./outputs/attacked_scaling_0.5.jpg", attacked)

# Wrapper function to apply different attacks
def apply_attack(image, attack_type="scaling", **kwargs):
    if attack_type == "scaling":
        return scaling(image, kwargs.get("scale_factor", 1.0))
    elif attack_type == "gaussian":
        return gaussian_noise(image, kwargs.get("mean", 0), kwargs.get("stddev", 10))
    elif attack_type == "spn":
        return salt_pepper_noise(image, kwargs.get("density", 0.05))
    elif attack_type == "jpeg":
        return jpeg_compression(image, kwargs.get("quality", 25))
    elif attack_type == "rotate":
        return rotation(image, kwargs.get("angle", 30))
    elif attack_type == "crop":
        return cropping(image, kwargs.get("crop_ratio", 0.8))
    elif attack_type == "brightness":
        return brightness_adjustment(image, kwargs.get("factor", 1.5))
    else:
        raise ValueError("Unknown attack type")

# Testing usage of the apply_attack function
# Apply JPEG compression
#attacked_jpeg = apply_attack(image, attack_type="jpeg", quality=15)
#cv2.imwrite("./utils/attacked_images/product7_attacked_jpeg.jpg", attacked_jpeg)

# Apply rotation
attacked_rotate = apply_attack(image, attack_type="rotate", angle=45)
cv2.imwrite("./utils/attacked_images/product9_attacked_rotate.jpg", attacked_rotate)

# Apply cropping
#attacked_crop = apply_attack(image, attack_type="crop", crop_ratio=0.7)
#cv2.imwrite("./utils/attacked_images/product7_attacked_crop.jpg", attacked_crop)

# Apply brightness adjustment
attacked_bright = apply_attack(image, attack_type="brightness", factor=2.0)
cv2.imwrite("./utils/attacked_images/product9_attacked_bright.jpg", attacked_bright)

# Apply Gaussian noise
attacked_gaussian = apply_attack(image, attack_type="gaussian", mean=0, stddev=25)
cv2.imwrite("./utils/attacked_images/product9_attacked_gaussian.jpg", attacked_gaussian)

# Apply salt-and-pepper noise
#attacked_spn = apply_attack(image, attack_type="spn", density=0.05) 
#cv2.imwrite("./utils/attacked_images/product7_attacked_spn.jpg", attacked_spn)

# Apply scaling
attacked_scale = apply_attack(image, attack_type="scaling", scale_factor=0.5)
cv2.imwrite("./utils/attacked_images/product9_attacked_scale.jpg", attacked_scale)