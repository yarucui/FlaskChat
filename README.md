# FlaskChat Project

FlaskChat is a simple and efficient implementation of a web-based chatbot built using Flask and a pre-trained machine learning model.

## Prerequisites

Before you begin, ensure you have the following installed on your system:

- Python (>= 3.7)
- pip (Python package installer)
- (Optional) Conda, if you plan to use `environment.yml`

## Setup Instructions

### Step 1: Clone the Repository

Clone the repository to your local machine:
```bash
git clone <repository_url>
cd <repository_name>
```

### Step 2: Set Up the Environment

Choose one of the following methods to set up your environment:

#### Method 1: Using Conda (Recommended)
If an `environment.yml` file is present, create and activate the environment:
```bash
conda env create -f environment.yml
conda activate chatbot_env
```

#### Method 2: Using pip
If `requirements.txt` is available, install dependencies using pip:
```bash
pip install -r requirements.txt
```

### Step 3: Train the Model

Run the `train.py` script to train the chatbot model. This will generate the following files:

- `words.pkl`
- `classes.pkl`
- `model.h5`

To train the model, execute:
```bash
python train.py
```

### Step 4: Start the Flask Application

Run the `flaskapp.py` file to start the chatbot web application:
```bash
python flaskapp.py
```

The application will start a web server and provide a URL (e.g., `http://127.0.0.1:5000/`). Open this URL in your web browser to interact with the chatbot.

## Notes

- Ensure all generated files (`words.pkl`, `classes.pkl`, `model.h5`) are in the same directory as `flaskapp.py`.
- The chatbot's behavior is defined by the training data in `intents.json`. Modify it to customize the responses and retrain the model.

## Troubleshooting

If you encounter any issues:

1. Verify that all dependencies are installed correctly.
2. Ensure you are using the correct version of Python.
3. Check for error messages in the terminal logs when running the scripts.
4. Confirm that the required files (`intents.json`, `words.pkl`, `classes.pkl`, `model.h5`) are present in the project directory.

