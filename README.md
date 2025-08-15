# Deepfake Detection

### Introduction

This repository contains the source code for a deepfake detection project. The goal of this project is to build a machine learning model capable of classifying and identifying deepfake videos.

### Project Structure

The project is organized into the following main directories:

-   **`configs/`**: Contains configuration files for the model and the training process.
-   **`data/`**: Contains scripts for data processing and potentially stores training datasets.
-   **`models/`**: Holds the model architecture definitions (e.g., `.py` files defining neural network models).
-   **`training/`**: Stores the main scripts for training, evaluating, and testing the model.
-   **`utils/`**: Contains utility functions and supporting tools for the project.
-   **`main.py`**: The primary entry point for running the project (training, evaluation, etc.).
-   **`requirements.txt`**: Lists the necessary Python libraries to run the project.

### Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/honglinhvt19/Deepfake_Detection.git](https://github.com/honglinhvt19/Deepfake_Detection.git)
    cd Deepfake_Detection
    ```

2.  **Install dependencies:**
    Ensure you have Python and pip installed.
    ```bash
    pip install -r requirements.txt
    ```

### Usage

To run the model training, you can use the `main.py` file. The exact syntax may vary, but it will typically follow a format like this:
```bash
python main.py --mode train --config configs/your_config_file.yaml
