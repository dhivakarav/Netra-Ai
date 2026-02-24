# NetraAI — Non-Line-of-Sight Presence Intelligence System

NetraAI is an intelligent surveillance system capable of detecting Anomaly presence even when the subject is not directly visible to the camera.

Unlike traditional computer vision systems that rely purely on object visibility, NetraAI infers presence using environmental interaction, motion continuity, and behavioural reasoning.

---

## Core Idea

Cameras normally answer:

> “Is someone visible?”

NetraAI answers:

> “Is someone present?”

The system detects hidden activity behind obstacles, in blind spots, and under poor visibility conditions such as fog, glare, or darkness.

---

## Key Features

* Blind-spot detection
* Hidden movement inference using shadow reasoning
* Behaviour anomaly detection
* Motion prediction using Kalman filtering
* Environmental awareness (fog, glare, blur, dark scenes)
* Explainable alerts instead of raw detections
* Real-time processing

---

## System Architecture

Camera → Object Detection → Shadow Reasoning → Tracking → Behaviour Analysis → Intelligent Alert

---

## Applications

* Smart surveillance systems
* Industrial safety monitoring
* Night-time perimeter security
* Restricted area monitoring
* Public infrastructure safety
* Search and rescue assistance

---

## Technology Stack

* Python
* OpenCV
* YOLOv8
* NumPy
* Kalman Filter
* Real-time video processing

---

## Example Output

Instead of:

> Motion detected

NetraAI reports:

> Person disappeared in blind zone and shadow persisted — suspicious activity

---

## Future Improvements

* Multi-camera coordination
* Depth reasoning
* Edge device deployment
* Predictive behaviour modelling

---

## Concept

NetraAI transforms cameras from recording devices into situational awareness systems.

---
