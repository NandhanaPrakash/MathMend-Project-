import os
import sys
from PIL import Image
import cv2
import numpy as np
import pytesseract
import easyocr


pytesseract.pytesseract.tesseract_cmd = r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe"  # <-- Change if needed


def preprocess_image(img_path):
    """
    Convert image to grayscale, denoise, and apply threshold
    """
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {img_path}")

    # Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Denoise
    denoised = cv2.fastNlMeansDenoising(gray, h=10)

    # Thresholding
    _, thresh = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Convert back to PIL Image for OCR engines
    return Image.fromarray(thresh)

# --- 3. Run Tesseract OCR ---
def run_tesseract(img):
    try:
        text = pytesseract.image_to_string(img, config="--oem 3 --psm 6")
        return text.strip()
    except Exception as e:
        return f"[Tesseract failed: {e}]"

# --- 4. Run EasyOCR ---
def run_easyocr(img_path):
    try:
        reader = easyocr.Reader(['en'], gpu=False)
        results = reader.readtext(img_path)
        text = " ".join([res[1] for res in results])
        return text.strip()
    except Exception as e:
        return f"[EasyOCR failed: {e}]"

# --- 5. Fuse results (Improved) ---
def fuse_text(tess_text, easy_text):
    # Check if Tesseract output is actually an error message
    if "Tesseract failed" in tess_text:
        return easy_text

    # Standard length check
    if len(tess_text) == 0 and len(easy_text) == 0:
        return ""
    if len(tess_text) >= len(easy_text):
        return tess_text
    return easy_text

# --- 6. Process single image ---
def process_image(img_path):
    preprocessed_img = preprocess_image(img_path)

    tess_text = run_tesseract(preprocessed_img)
    easy_text = run_easyocr(img_path)

    print(f"\n--- {os.path.basename(img_path)} ---")
    print("\nTesseract Output:\n", tess_text)
    print("\nEasyOCR Output:\n", easy_text)

    fused = fuse_text(tess_text, easy_text)
    print("\nFused Output:\n", fused)
    print("---------------------------")

    return fused

# --- 7. Process folder ---
def process_folder(folder_path, save_results=False, output_folder="ocr_results"):
    if not os.path.exists(folder_path):
        print(f"ERROR: Folder not found: {folder_path}")
        return

    if save_results:
        os.makedirs(output_folder, exist_ok=True)

    supported_ext = (".png", ".jpg", ".jpeg", ".bmp", ".tiff")
    images = [f for f in os.listdir(folder_path) if f.lower().endswith(supported_ext)]
    if not images:
        print(f"No images found in folder: {folder_path}")
        return

    for img_file in images:
        img_path = os.path.join(folder_path, img_file)
        fused_text = process_image(img_path)
        if save_results:
            out_path = os.path.join(output_folder, os.path.splitext(img_file)[0] + ".txt")
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(fused_text)
    print("\n✅ All images processed.")


if __name__ == "__main__":
    fixed_path = r"C:\\Users\\Dell\\Downloads\\i hope its this\\MathMend-final_stable\\mathq2.png"
    
    # Set this to True if you want to save the text to a file, else False
    save_flag = False 

    print(f"Processing: {fixed_path} ...")

    if os.path.isfile(fixed_path):
        # It's a single file
        process_image(fixed_path)
    elif os.path.isdir(fixed_path):
        # It's a folder of images
        process_folder(fixed_path, save_results=save_flag)
    else:
        print(f"ERROR: The path does not exist: {fixed_path}")
        print("Check if the file name is correct or if you forgot the file extension (like .png)")