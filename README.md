🧤 Glove Detection in Manufacturing: Ablation Study on YOLOv8m-seg
🚀 Overview
Welcome to the official repository for our research on lightweight glove/no-glove detection in manufacturing environments, based on systematic ablation studies of the YOLOv8m-seg (medium) segmentation model!

This project investigates how architectural modifications to YOLOv8m-seg affect speed, accuracy, and deployability for real-time PPE compliance monitoring on edge devices.

🏗️ What’s Inside?
📚 Detailed ablation experiments on backbone, neck, and segmentation heads

🔬 Performance comparison between baseline and progressively simplified models

⚡ Open-sourced minimal architectures for edge deployment (Jetson, etc.)

🗂️ Data splits, metrics, and training configurations for reproducibility

✨ Key Features
Systematic Ablations:

Backbone/neck depth & width reduction

SPPF (Spatial Pyramid Pooling) removal

Deepest (P5) feature path ablation

Ultra-compact segmentation heads: P3-only mask prediction

Optimized for Small Objects:

Tailored for detecting gloves and hand PPE in complex, real-world assembly lines

Lightweight Edge Deployment:

Substantial reduction in parameters & GFLOPs (from 110.0 → as low as 10.0)

Models tested on Jetson Nano, Orin, and desktop GPUs

📝 Quick Results Table
Model Variant	mAP (%)	FPS	Params (M)	GFLOPs	Notes
Baseline	—	—	27.3	110.0	YOLOv8m-seg
Backbone Lite	—	—	—	~65	Reduced BB
SPPF Removed	—	—	—	~60	No SPPF
P5 Removed	—	—	—	~39	No P5 (P3/P4)
P3 Only	—	—	—	~22	P3 head only
(Fill in mAP/FPS with your results)

📦 Experimental Settings
Parameter	Value
Epochs	100
Batch Size	16
Learning Rate	0.01
Optimizer	AdamW
Weight Decay	0.0005
Device	CUDA 12.4
GPU	NVIDIA TITAN RTX (24GB)
OS	Windows 11 Pro
Early Stopping	Patience = 30 epochs
