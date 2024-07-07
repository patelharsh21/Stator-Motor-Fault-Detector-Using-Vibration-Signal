
## Overview

The AIFD MotorFault GUI is a graphical user interface designed to aid in the intelligent diagnosis of stator faults in induction motors. This project leverages vibration signals and machine learning to provide an end-to-end solution for motor fault detection and severity prediction. The GUI allows users to upload motor data and receive detailed fault diagnosis results without requiring any coding knowledge.
App link: [AIFD MotorFault GUI](https://stator-motor-deploy-app-amybcmrxux5fiyg5pdsa9t.streamlit.app/)
Paper link : [Research Paper](https://github.com/patelharsh21/stator-motor-deploy-streamlit/blob/main/photos/AIFD2023_MotorFault_GUI.pdf)
## Features

- **User-Friendly Interface:** Built with Streamlit, the GUI enables easy dataset uploads, exploratory data analysis (EDA) visualization, and interaction with prediction results.
- **Automated Model Selection:** Utilizes AutoKeras to automatically select and optimize the best deep learning model for the given dataset.
- **Explainable AI:** Incorporates Explainable AI techniques to improve model transparency and trustworthiness.
- **High Accuracy:** Achieves 99.81% accuracy in stator fault detection and severity prediction.

## Requirements

- Python 3.7+
- Streamlit
- AutoKeras
- scikit-learn
- pandas
- numpy
- matplotlib

## Installation

1. Clone the repository:

   ```bash
   git clone git@github.com:patelharsh21/stator-motor-deploy-streamlit.git
   ```

2. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Run the Streamlit application:

   ```bash
   streamlit run app.py
   ```

2. Open your web browser and go to `http://localhost:8501`.

3. Upload your motor vibration data through the provided interface. A sample vibration data file is provided in the repo.
4. View the EDA results and interact with the fault diagnosis predictions.

### Block Diagram 
![Block Diagram](https://github.com/patelharsh21/stator-motor-deploy-streamlit/blob/main/photos/Pasted%20image.png)

### Methodology
1. **Data Collection:** Vibration signals from the stator motor are collected using a Brüel & Kjær (B&K) accelerometer.
2. **Preprocessing:** The data undergoes preprocessing, including invalid sample removal and normalization.
3. **Feature Extraction:** Various time-domain and frequency-domain features are extracted from the vibration data.
4. **Model Training:** An AutoKeras model is trained on the processed data to identify the optimal model for fault diagnosis.
5. **Prediction and Explanation:** The trained model predicts stator faults and their severity, with Explainable AI providing insights into the model's decision-making process.

## Results

### UI

![UI](https://github.com/patelharsh21/stator-motor-deploy-streamlit/blob/main/photos/UI%20(1).pdf)

### Results

![Confusion matrix](https://github.com/patelharsh21/stator-motor-deploy-streamlit/blob/main/photos/confusionMatrix.pdf)

![Classification metrics](https://github.com/patelharsh21/stator-motor-deploy-streamlit/blob/main/photos/metrics.pdf)

### XAI (Explainable AI)
![Summary plot](https://github.com/patelharsh21/stator-motor-deploy-streamlit/blob/main/photos/summaryPlot.pdf)

The proposed method demonstrates high accuracy in detecting and predicting the severity of stator faults in induction motors. The integration of a user-friendly GUI and Explainable AI techniques ensures that the solution is both accessible and trustworthy for end-users in industrial applications.

## Contributors

- Jitendra Kumar Dewangan (IIIT Naya Raipur)
- Harsh Patel (IIIT Naya Raipur)
- Aparna Sinha (IIIT Naya Raipur)
- Debanjan Das (IIT Kharagpur)


