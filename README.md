# Plant-Disease-Detection-System

This project aims to develop an intelligent system capable of **accurately detecting plant diseases** from images of crop leaves using **deep learning techniques**. Leveraging a large-scale dataset from **Kaggle** and a **CNN-based model built on MobileNetV2**, the system classifies plant leaf images into **38 distinct categories**, including both disease types and healthy conditions.

The goal is to assist farmers and agricultural professionals in early identification of plant diseases, thereby reducing crop loss and improving yield quality through timely intervention.

Dataset source: https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset?resource=download

Model Architecture: MobileNetV2

We use **MobileNetV2**, a highly efficient CNN architecture designed for mobile and real-time inference. Its lightweight structure makes it ideal for deployment on low-resource devices without compromising accuracy.

Key model layers:
- Pretrained MobileNetV2 base (`imagenet` weights)
- Global Average Pooling
- Dense (128 units, ReLU)
- Dense (38 units, Softmax)
