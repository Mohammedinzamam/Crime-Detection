**Weapon and Knife Detection System**
This repository provides the codebase and assets for an academic project focused on identifying weapons, specifically firearms and knives, using advanced computer vision techniques. The work was carried out as part of research in the field of cybersecurity.

**Project Summary**
The goal of this project is to create and deploy a smart, automated system capable of detecting guns and knives in real time. By leveraging the YOLOv8 (You Only Look Once) deep learning model along with transfer learning methods, the system aims to enhance public safety through constant video monitoring.

**Main Highlights**
YOLOv8-Based Detection: Utilizes the YOLOv8 architecture for fast and accurate object detection in live video streams.

Transfer Learning Implementation: Employs transfer learning to fine-tune the model for weapon detection, improving its performance in the intended use case.

IP Camera Compatibility: Designed for smooth integration with IP cameras, enabling real-time monitoring and rapid alert generation.

**Dataset**
Dataset link: [Weapon 2 Dataset](https://universe.roboflow.com/joao-assalim-xmovq/weapon-2/dataset/2)

**Usage Instructions**
**Clone the repository:**

git clone https://github.com/JoaoAssalim/Weapons-and-Knives-Detector-with-YOLOv8.git

**Install the required dependencies:**

pip install -r requirements.txt

**Run the system:**
**For processing stored images:**

python detecting-images.py

**For real-time detection from a live camera feed:**

python live_detection.py
