# Medical Text Classification with FastAPI and BERT

This project demonstrates a simple web application for medical text classification using FastAPI and BERT (Bidirectional Encoder Representations from Transformers). The web application allows users to input medical text, and the BERT model predicts the corresponding class.

## Repository cloning to local environment

Please run the below command to clone the repository to local and then change directory as mentioned below

git clone https://github.com/Shivamjohri247/med-text-classify.git
cd med-text-classify

## Model weights

Model weights are hosted in the path below which needs to be downloaded and stored in model_artifacts folder

path - https://drive.google.com/file/d/1dElK3DJL60ta6gsGhMwhBbG0_DFYBJM5/view?usp=drive_link

## Prerequisites

Before running the application, ensure that you have the necessary dependencies installed. You can install them using the following command:

pip install -r requirements.txt

After successful completion of requirements installation, please run the command below to start the inference script and navigate to http://localhost:8000

uvicorn inference:app --port 8000 --reload



# Notes

This application uses a pre-trained BERT model for sequence classification. You can replace the model with your own trained model if needed.

Customize the HTML file (webpage.html) as per your design requirements.

For production use, consider deploying the FastAPI application using a production-ready ASGI server, such as Gunicorn.

