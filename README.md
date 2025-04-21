# Iris_MLflow
Jupyter Notebook demonstrating Decision Tree Model training on Iris dataset with MLflow.
# Iris Classification with Decision Tree and MLflow Tracking

## What the project does

This project demonstrates a basic machine learning workflow using Python and common data science libraries. It involves:

1.  **Loading** the classic Iris dataset from `scikit-learn`.
2.  **Preprocessing** the data slightly using `pandas`.
3.  **Splitting** the data into training and testing sets.
4.  **Training** a `DecisionTreeClassifier` model from `scikit-learn` to classify the Iris species based on their features.
5.  **Evaluating** the model's performance using metrics like accuracy, precision, recall, F1-score, and visualizing a confusion matrix with `seaborn` and `matplotlib`.
6.  **Tracking** the experiment using **MLflow**. This includes logging:
    * Model hyperparameters (like `max_depth`, `random_state`).
    * Performance metrics.
    * The confusion matrix plot as an artifact.
    * The trained `scikit-learn` model itself.
7.  **Loading** the saved model back from the MLflow run.
8.  **Using** the loaded model to make predictions on the test data to verify it works.

The goal is to show a simple yet complete cycle of training, evaluating, tracking, saving, and reloading a machine learning model.

## Requirements

* **Python:** Version 3.7 or higher is recommended.
* **Libraries:** You'll need the following Python libraries:
    * `scikit-learn` (for the dataset, model, metrics, and splitting)
    * `mlflow` (for experiment tracking and model logging/loading)
    * `pandas` (for data manipulation)
    * `numpy` (used by pandas/sklearn)
    * `seaborn` (for plotting the confusion matrix)
    * `matplotlib` (for plotting)
    * `jupyterlab` or `notebook` (to run the `.ipynb` file)

## How to Run the Notebook

1.  **Clone the Repository (Optional):**
    ```bash
    git clone <your-repository-url>
    cd <your-repository-name>
    ```
    Alternatively, just download the `.ipynb` file.

2.  **Create a Virtual Environment (Recommended):**
    Open your terminal or command prompt in the project directory.
    ```bash
    # Create the environment (e.g., named 'venv')
    python -m venv venv
    # Activate the environment
    # On Windows:
    .\venv\Scripts\activate
    # On macOS/Linux:
    source venv/bin/activate
    ```

3.  **Install Requirements:**
    ```bash
    pip install scikit-learn mlflow pandas numpy seaborn matplotlib jupyterlab
    ```
    *(Optional: If a `requirements.txt` file is provided, you can use `pip install -r requirements.txt`)*

4.  **Start the MLflow Tracking UI:**
    For MLflow to log your experiments and models, you need a tracking server running. For local development, you can run the MLflow UI:
    ```bash
    mlflow ui
    ```
    This usually starts the server at `http://127.0.0.1:5000`. Keep this running in a separate terminal while you run the notebook. *Note: The notebook code should have `mlflow.set_tracking_uri("http://127.0.0.1:5000/")` (or your server's address) to connect to this UI.*

5.  **Run Jupyter:**
    ```bash
    jupyter lab
    # OR
    # jupyter notebook
    ```
    This will open the Jupyter interface in your web browser.

6.  **Open and Run the Notebook:**
    Navigate to and open the `.ipynb` notebook file within the Jupyter interface. Run the cells sequentially.

## MLflow Integration

This project heavily utilizes **MLflow** for MLOps (Machine Learning Operations) practices:

* **Experiment Tracking:** Every time you run the notebook sections related to training and logging, a new "run" is created within the "iris_decision_tree" experiment (or the name set in the notebook).
* **Logging:** The following are logged to the MLflow run:
    * `log_param`: Key hyperparameters of the Decision Tree.
    * `log_metric`: Performance scores (accuracy, precision, etc.).
    * `log_artifact`: The confusion matrix image file.
    * `log_model`: The complete trained `sklearn` model, packaged by MLflow.
* **Viewing Results:** Open your MLflow UI (typically `http://127.0.0.1:5000` as started in step 4) in your browser. You can explore the experiment runs, compare metrics/parameters across different runs, and view the logged artifacts (like the plot and the model files).
* **Model Loading:** The final part of the notebook demonstrates how to load the model directly from a specific MLflow run using its Run ID and the model artifact path, showcasing how a model logged by MLflow can be easily retrieved for later use (e.g., for deployment or inference).

**Feel free to experiment with different model parameters in the notebook and re-run it – you'll see new runs appear in the MLflow UI, making it easy to track your experiments!**
