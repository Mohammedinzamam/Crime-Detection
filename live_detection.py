import cv2
import requests
from ultralytics import YOLO
from datetime import datetime
import smtplib
from email.message import EmailMessage
import os

# --- Configuration ---
YOLO_MODEL_PATH = './runs/detect/Normal_Compressed/weights/best.pt'
SENDER_EMAIL = "inzamam8055@gmail.com"
SENDER_APP_PASSWORD = "afgn dlvp falm bvvb"
RECEIVER_EMAIL = "intham8055@gmail.com"

# Minimum confidence threshold for reporting
CONF_THRESHOLD = 0.5

# --- Load YOLOv8 model ---
model = YOLO(YOLO_MODEL_PATH)

# --- Fetch approximate camera location via IP geolocation ---
def fetch_location():
    try:
        resp = requests.get("https://ipinfo.io", timeout=5)
        data = resp.json()
        loc = data.get("loc", "")  # "lat,long"
        city = data.get("city", "")
        region = data.get("region", "")
        country = data.get("country", "")
        return loc, city, region, country
    except Exception:
        return "", "", "", ""

# Generate Google Maps link if we have lat/long
def maps_link(latlong):
    return f"https://maps.google.com?q={latlong}" if latlong else "Location unavailable"

# --- Email sending with image + metadata ---
def send_report(image_path, timestamp, loc_str, human_location, map_url):
    msg = EmailMessage()
    msg["From"] = SENDER_EMAIL
    msg["To"] = RECEIVER_EMAIL
    msg["Subject"] = f"🚨 Crime Detected at {timestamp}"

    body = (
        f"**Crime incident detected**\n\n"
        f"Time (local): {timestamp}\n"
        f"Approximate Location: {human_location}\n"
        f"Map: {map_url}\n\n"
        f"Please review the attached image."
    )
    msg.set_content(body)

    # Attach the image snapshot
    with open(image_path, "rb") as img_file:
        img_data = img_file.read()
        img_name = os.path.basename(image_path)
    msg.add_attachment(img_data, maintype="image", subtype="jpeg", filename=img_name)

    # Send via SMTP
    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
            smtp.login(SENDER_EMAIL, SENDER_APP_PASSWORD)
            smtp.send_message(msg)
        print(f"✅ Email report sent: {timestamp}")
    except Exception as e:
        print(f"❌ Failed to send email: {e}")

# --- Main live detection loop ---
def live_detection():
    cap = cv2.VideoCapture(0)  # Replace with your camera stream if needed
    if not cap.isOpened():
        print("Error: Cannot access camera.")
        return

    latlong, city, region, country = fetch_location()
    loc_str = f"{latlong}" if latlong else "Unavailable"
    human_loc = f"{city}, {region}, {country}" if city else "Unavailable"
    map_url = maps_link(latlong)

    print(f"Camera location approx: {human_loc} ({loc_str})")
    print("Press 'q' to stop detection.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, conf=CONF_THRESHOLD, verbose=False)
        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                label = r.names[cls_id].lower()
                conf = float(box.conf[0])

                if label in ("gun", "knife"):
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    fname = f"incident_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
                    cv2.imwrite(fname, frame)
                    print(f"Detected {label} at {timestamp}, sending report...")

                    send_report(fname, timestamp, loc_str, human_loc, map_url)

        cv2.imshow("Live Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Stopping detection.")
            break

    cap.release()
    cv2.destroyAllWindows()

# --- Run ---
if __name__ == "__main__":
    live_detection()
