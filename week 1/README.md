
### NOTE
> Thefile project contains code to detrmine model to use
> the main.py hass the whole model in one script 
> full_project contains the code in a note book 


steps to Create a machine learning model tha predicts whether a custor is likely to leave the company or not 


# Step 1: Problem Understanding and Data Collection

    Understand the business problem: its  very clear from the question that predicting customer churn is the problem to be solved.
    
    Collect relevant data:which may directly or indirectly influence churn  and churn labels (whether a customer has churned or not).

# Step 2: Data Cleaning and Preprocessing

    Data Cleaning: Handle missing values, duplicates, and outliers to ensure the quality of the dataset.
    For me the data i got from kagle and it doesn`t  have missing values

# Step 3: Feature Engineering

    Create relevant features: Generate new features or extract meaningful information from existing ones that might be predictive of churn.

    Feature Selection: Identify the most important features using techniques like feature importance or correlation analysis.
    This can be achive my manual check of features which affect customer churn or by use of feature decomposition or bu use of Principal Component Analysis (PCA)
    this will reduce number of features to a desired amount while still selecting the relevant ones

    In this project i tried usiong the relevant columns listed from in the documentation for this dataset but its perfromance was poor compared to 
    encoding all the columns adn using pca to reduce the used columns to 12
 
# Step 4: Data Splitting

    Split the dataset into training, validation, and test sets.
    I did the spliting manually by index 9000 for training and valuidation and the rest 1000 for testing.

# Step 5: Model Selection and Training

    Choose suitable machine learning algorithms for classification. Common choices include logistic regression, decision trees, random forests, gradient boosting, and neural networks.
    For this project i used a neral networks.
    Train multiple models with various hyperparameters to find the best-performing one.
    Evaluate models using appropriate metrics (accuracy, precision, recall, F1-score, ROC AUC) on the validation set.

# Step 6: Model Evaluation and Tuning

    Fine-tune hyperparameters using techniques like grid search or random search to improve model performance.
    Use techniques like cross-validation to ensure the model's generalization ability.

# Step 8: Model Deployment

    Deploy the best-performing model into a production environment, ensuring that it can handle real-ti