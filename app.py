import streamlit as st
import numpy as np
import cv2
from ultralytics import YOLO

# Streamlit app configuration
st.set_page_config(
    page_title="Plant Disease Detector with Recommendations", page_icon="🌿")

st.title("Plant Disease Detector with Recommendations")
st.markdown(
    "This application detects plant diseases using YOLOv8 and provides recommendations.")

# Function to convert file buffer to OpenCV image


def create_opencv_image_from_buffer(img_buffer, cv2_img_flag=1):
    try:
        img_buffer.seek(0)
        img_array = np.asarray(bytearray(img_buffer.read()), dtype=np.uint8)
        return cv2.imdecode(img_array, cv2_img_flag)
    except Exception as e:
        st.error(f"Error converting image: {e}")
        return None

# Load YOLO model


@st.cache_resource
def load_yolo_model(weights_path):
    return YOLO(weights_path)


# Load trained YOLOv8 model (replace with your custom-trained model path)
YOLO_WEIGHTS = "Runs/best.pt"  # Update this with the correct path
model = load_yolo_model(YOLO_WEIGHTS)

# Class names
class_names = [
    "Apple__Apple_scab", "Apple_Black_rot", "Apple_Cedar_apple_rust", "Apple__healthy",
    "Blueberry__healthy", "Cherry_Powdery_mildew", "Cherry__healthy",
    "Corn__Cercospora_leaf_spot_Gray_leaf_spot", "Corn_Common_rust", "Corn__Northern_Leaf_Blight",
    "Corn__healthy", "Grape_Black_rot", "Grape_Esca(Black_Measles)",
    "Grape__Leaf_blight(Isariopsis_Leaf_Spot)", "Grape___healthy",
    "Orange__Haunglongbing(Citrus_greening)", "Peach__Bacterial_spot", "Peach__healthy",
    "Pepper,bell_Bacterial_spot", "Pepper,_bell__healthy",
    "Potato__Early_blight", "Potato_Late_blight", "Potato__healthy",
    "Raspberry__healthy", "Soybean_healthy", "Squash__Powdery_mildew",
    "Strawberry__Leaf_scorch", "Strawberry__healthy",
    "Tomato__Bacterial_spot", "Tomato_Early_blight", "Tomato__Late_blight",
    "Tomato__Leaf_Mold", "Tomato__Septoria_leaf_spot",
    "Tomato__Spider_mites_Two-spotted_spider_mite", "Tomato__Target_Spot",
    "Tomato__Tomato_Yellow_Leaf_Curl_Virus", "Tomato_Tomato_mosaic_virus", "Tomato__healthy"
]

# Recommendations for each class


def get_recommendation(class_id, class_name):
    recommendations = {
        0: f"Treat {class_name} using recommended fungicide and remove fallen leaves.",
        1: f"Remove affected branches of {class_name} and apply copper spray.",
        2: f"Manage {class_name} by planting resistant varieties.",
        3: f"{class_name} is healthy, no action required.",
        4: f"{class_name} is healthy, no action required.",
        5: f"Spray sulfur-based fungicides for {class_name}.",
        6: f"{class_name} is healthy, no action required.",
        7: f"Treat {class_name} with crop rotation and resistant hybrids.",
        8: f"Apply fungicide to manage {class_name}.",
        9: f"Use crop rotation and fungicides to manage {class_name}.",
        10: f"{class_name} is healthy, no action required.",
        11: f"Remove infected parts of {class_name} and apply fungicide.",
        12: f"Prune and remove infected plants for {class_name}.",
        13: f"Treat {class_name} by applying Bordeaux mixture.",
        14: f"{class_name} is healthy, no action required.",
        15: f"Manage {class_name} by controlling psyllid population.",
        16: f"Use fixed copper sprays to treat {class_name}.",
        17: f"{class_name} is healthy, no action required.",
        18: f"Control {class_name} with copper-based sprays.",
        19: f"{class_name} is healthy, no action required.",
        20: f"Apply fungicide to manage {class_name}.",
        21: f"Remove affected foliage and apply fungicides for {class_name}.",
        22: f"{class_name} is healthy, no action required.",
        23: f"{class_name} is healthy, no action required.",
        24: f"{class_name} is healthy, no action required.",
        25: f"Spray potassium bicarbonate for {class_name}.",
        26: f"Remove infected leaves and improve air circulation for {class_name}.",
        27: f"{class_name} is healthy, no action required.",
        28: f"Apply copper-based bactericides for {class_name}.",
        29: f"Prune and apply fungicides to control {class_name}.",
        30: f"Remove affected plants and use resistant varieties for {class_name}.",
        31: f"Improve air circulation and apply fungicides for {class_name}.",
        32: f"Use crop rotation and fungicides to treat {class_name}.",
        33: f"Use miticides or neem oil for {class_name}.",
        34: f"Apply fungicides and crop rotation for {class_name}.",
        35: f"Control whiteflies with insecticides for {class_name}.",
        36: f"Disinfect tools and hands to prevent the spread of {class_name}.",
        37: f"{class_name} is healthy, no action required.",
    }
    return recommendations.get(class_id)


# File uploader for images
img_files = st.file_uploader("Upload plant images", type=[
                             'png', 'jpg', 'jpeg'], accept_multiple_files=True)

if img_files:
    for n, img_file_buffer in enumerate(img_files):
        if img_file_buffer:
            # Convert the uploaded file to OpenCV image
            open_cv_image = create_opencv_image_from_buffer(img_file_buffer)
            if open_cv_image is not None:
                st.subheader(f"Image {n + 1}")
                st.image(open_cv_image, channels="BGR",
                         caption=f"Original Image {n + 1}")

                # Perform YOLO inference
                results = model.predict(open_cv_image, conf=0.25)
                detections = results[0]  # Single image results

                # Annotate image
                annotated_image = detections.plot()

                # Display the annotated image
                st.image(annotated_image, channels="BGR",
                         caption=f"Detection Result {n + 1}")

                # Extract detected classes and confidence scores
                detected_classes = [
                    class_names[int(cls)] for cls in detections.boxes.cls.cpu().numpy()]
                confidence_scores = detections.boxes.conf.cpu().numpy()

                st.markdown("### Detection Results:")
                # In the detection loop
                if detected_classes:
                    for disease, score in zip(detected_classes, confidence_scores):
                        st.write(f"- **{disease}**: {score:.2%} confidence")
                        # Get and display the recommendation
                        recommendation = get_recommendation(
                            detected_classes.index(disease), disease)
                        st.write(f"  - Recommendation: {recommendation}")
                else:
                    st.write("No diseases detected.")
