📌 Problem Statement

Brain tumors, whether benign or malignant, pose life-threatening risks if not detected early. Increasing intracranial pressure can cause brain damage and severe complications. Hence, early and accurate tumor detection is crucial for saving lives.

🎯 Objective

The goal is to develop a reliable, efficient, and explainable AI system that:

Detects and classifies brain tumors from MRI scans.

Improves diagnostic accuracy while reducing human error.

Provides interpretable visual explanations (Grad-CAM++).

Generates medical-style reports with treatment suggestions.

Can be deployed in real-world healthcare settings.

👥 Target Users

Doctors & Radiologists → Faster, AI-assisted diagnosis

Hospitals & Diagnostic Centers → Improved efficiency & reduced costs

Oncologists → Early detection aids better treatment planning

Medical Researchers → Tool for advancing medical AI research

Patients → Indirect beneficiaries with timely treatment

📂 Dataset

Source: Kaggle – Brain Tumor MRI Dataset (Masoud Nickparvar)

Total Images: 7,033 MRI scans

Classes:

Glioma

Meningioma

Pituitary

No Tumor

Class	Training Images	Testing Images
Glioma	1321	300
Meningioma	1339	306
Pituitary	1457	300
No Tumor	1595	405
🏗️ Proposed System

Hybrid CNN (ResNet-18) + Vision Transformer (ViT)

Grad-CAM++ for tumor region visualization

Segmentation Models: Thresholding, Otsu’s method, Watershed

3D Tumor Visualization with quantitative metrics

LLM Integration (Ollama Gemma-2B) → Generates medical-style reports and treatment suggestions

Automation (n8n) → Automates workflows and report delivery

⚙️ System Architecture

Input Layer → User uploads MRI scans

Preprocessing → Normalization, resizing, augmentation

Deep Learning Model → ResNet-18 + ViT hybrid

Segmentation & Visualization → Tumor detection & heatmaps (Grad-CAM++)

Post-Processing → Report generation via LLM

Automation & Delivery → Email delivery to doctors/patients

🚀 Tech Stack

AI/ML Models: ResNet-18, Vision Transformer, DenseNet-121, Swin Transformer, Custom CNN

Explainable AI: Grad-CAM++

LLM: Ollama Gemma-2B

Backend: Flask

Frontend: HTML, CSS, JavaScript

Database: PostgreSQL

Automation: n8n

Visualization: 2D overlays & interactive 3D rendering

🧪 Results

Achieved accurate classification of brain tumors using hybrid CNN+ViT.

Generated interpretable heatmaps for clinical trust.

Automated medical-style reporting with treatment recommendations.

Lightweight system: Runs on CPU laptops (no GPU required).

📊 Innovations

Hybrid CNN + Transformer architecture for local + global feature extraction.

Explainable AI with Grad-CAM++ for trustworthy results.

Medical Report Generation using LLM.

Workflow Automation with n8n.

3D Visualization for tumor volume & shape analysis.

👨‍💻 Team Contributions
Member	Contribution
Guruprasath P	Ensemble Model (EfficientNet-B7 + ViT), n8n Automation
Harish R R	Swin Transformer, 3D Visualization
Harsavardhini R	Hybrid CNN+ViT, LLM Integration
Harsitha R	DenseNet-121
Jayanesh D	Custom CNN
Karthik A	Hybrid Transformer, BioGPT-3
Kaviya S	ResNet-18, LLM Integration
Keerthevasan T S	XceptionNet, Frontend, Backend
Keerthika P	ResNet-18 + CBAM, Architecture Design
📜 Mentors

Vipin Singhal (CTS Mentor)

Renuka Devi S (College Mentor – Rajalakshmi Engineering College)

⚖️ License

This project is intended for research and educational purposes only. Not approved for direct clinical use.
