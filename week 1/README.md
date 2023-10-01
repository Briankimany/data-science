
### NOTE

The main.py hass the whole model in one script 

Full_project contains the code in a note book 


steps to Create a machine learning model tha predicts whether a custor is likely to leave the company or not 


### Step 1: Problem Understanding and Data Collection

The objective of this project is to develop a machine learning model that predicts whether customers are likely to churn (leave) a telecommunications company based on historical data, enabling the company to proactively address customer retention strategies.

Collect relevant data:which may directly or indirectly influence churn  and churn labels (whether a customer has churned or not).
The data is collected from 

[kaggle] (https://www.kaggle.com/datasets/blastchar/telco-customer-churn/download?datasetVersionNumber=1)

Its a fataset showing the churn rate of certain bank

### Step 2: Data Cleaning and Preprocessing

Data Cleaning: Handle missing values, duplicates, and outliers to ensure the quality of the dataset.
The dataset from Kaggle is assumed to be clean with no missing values.However, it's important to verify this by checking for null values

```python 
# load the data from local storage
data = pd .read_csv("path to data")

# cheking for missing  values
data.describe()


```

### Step 3: Feature Engineering

Create relevant features: Generate new features or extract meaningful information from existing ones that might be predictive of churn.

Feature Selection: Identify the most important features using techniques like feature importance or correlation analysis.
This can be achive my manual check of features which affect customer churn or by use of feature decomposition or bu use of Principal Component Analysis (PCA)
this will reduce number of features to a desired amount while still selecting the relevant ones

In this project i tried using the relevant columns listed from in the documentation for this dataset but its perfromance was poor compared to 
encoding all the columns and using pca to reduce the target columns to 12

```python 
# Feature generation
data = pd.get_dummies(data)


# scale the data  and split it 
def Scaler(new_data , ratio = 9000):
    train_data = new_data[:ratio]
    test_data = new_data[ratio:]
    train_data_val = train_data.pop('Exited')
    test_data_val = test_data.pop("Exited")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(train_data.to_numpy())
    Y_train = train_data_val.to_numpy()
    X_test_scaled = scaler.fit_transform(test_data.to_numpy())
    Y_test = test_data_val.to_numpy()
    return X_train_scaled , Y_train , X_test_scaled , Y_test

## scale the data 
X_train , Y_train , X_test  , Y_test = Scaler(full_data)

## dimensionality reduction
def pca_trans(X_train , X_test , n_comp = 12):
    from sklearn.decomposition import PCA
    shape_1 = X_train.shape
    pca = PCA(n_components=n_comp)
    pca = pca.fit(X_train)

    X_train_pca = pca.transform(X_train)
    X_test_pca = pca.transform(X_test)
    shape_2 = X_train_pca.shape
    print (f"INPUT DATA SHAPE WAS {shape_1}  final shape = {shape_2}")
    return X_train_pca , X_test_pca
X_train_pca , X_test_pca = pca_trans(X_train , X_test , n_comp = 12)

```

### Step 5: Model Selection and Training



Choose suitable machine learning algorithms for classification.
Common choices include logistic regression, decision trees, random forests, gradient boosting, and neural networks.


For this project i used a neral networks.
This is due to that they woked well with this data set and produced an Area Under the Curve value(AUC) that was better compared to LR AND Randomforest
```python 

def neural_net(X_train_scaled , y_train , PLOT = False , num_epochs = 80 , input_shape =12 , train = True):
    # Build the neural network
    model = keras.Sequential([
        keras.layers.Dense(256, activation='relu', input_shape=(input_shape,), kernel_regularizer=regularizers.l2(0.01)),
        keras.layers.Dropout(0.4),  # Dropout layer with a 50% dropout rate
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dropout(0.4),
        keras.layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
        keras.layers.Dropout(0.4),
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dropout(0.4),  # Dropout layer with a 50% dropout rate
        keras.layers.Dense(1, activation='sigmoid')  # Binary classification
    ])

    model.compile(optimizer=keras.optimizers.Adam(0.001), loss='binary_crossentropy', metrics=['accuracy'])
    
    checkpoint = ModelCheckpoint(f'best_model_{num_epochs}.h5', monitor='val_loss', save_best_only=True)

    # Training
    if train :
        history = model.fit(X_train_scaled, y_train, epochs=num_epochs, batch_size=32, validation_split=0.2 ,  callbacks=[checkpoint])
    else:
        
        return print((model.summary()))
    if PLOT:
        training_loss = history.history['loss']
        training_accuracy = history.history['accuracy']
        validation_loss = history.history['val_loss']
        validation_accuracy = history.history['val_accuracy']

        # Plot the training and validation loss
        plt.plot(training_loss, label='Training Loss')
        plt.plot(validation_loss, label='Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()

        # Plot the training and validation accuracy
        plt.plot(training_accuracy, label='Training Accuracy')
        plt.plot(validation_accuracy, label='Validation Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.show()

   
    return model
    
# Its in a funtion  neural_net(X_train_scaled , y_train , PLOT = False , num_epochs = 80 , input_shape =12 , train = True)

# This funtion returns a trained model

```

I would train multiple models with various hyperparameters to find the best-performing one.

For this model it worked better on epochs ranging from 70 - 150 and batch size 32 with the input shape of 12

Fine-tune hyperparameters using techniques like grid search or random search to improve model performance.

### Step 6: Model Evaluation 

To evaluate the model, I used a custom function called `compute_metrics`, which:
- Plots the ROC curve, or receiver operating characteristic curve.
- Also, the confusion matrix for the classifications.

Getting the models prediction  and printing the accuracy value
```python 
y_pred = model.predict(X_test_pca)

_, acc = model.evaluate(X_test_pca, Y_test)
print("Accuracy = ", (acc * 100.0), "%")
```

Determining the threshold value

```python


fpr, tpr, thresholds = roc_curve(Y_test, y_pred)

i = np.arange(len(tpr)) 
roc = pd.DataFrame({'tf' : pd.Series(tpr-(1-fpr), index=i), 'thresholds' : pd.Series(thresholds, index=i)})
ideal_roc_thresh = roc.iloc[(roc.tf-0).abs().argsort()[:1]]  #Locate the point where the value is close to 0
print("Ideal threshold is: ", ideal_roc_thresh['thresholds']) 

```
visualizing the models performance

```python 


def compute_metrics(y_pred ,Y_test , thresh ):
    y_pred_binary = y_pred >= thresh
    fpr, tpr, thresholds = roc_curve(Y_test, y_pred)

    i = np.arange(len(tpr)) 
    roc = pd.DataFrame({'tf' : pd.Series(tpr-(1-fpr), index=i), 'thresholds' : pd.Series(thresholds, index=i)})
    ideal_roc_thresh = roc.iloc[(roc.tf-0).abs().argsort()[:1]]  #Locate the point where the value is close to 0
    print("Ideal threshold is: ", ideal_roc_thresh['thresholds']) 

    from sklearn.metrics import auc
    auc_value = auc(fpr, tpr)
    print("Area under curve, AUC = ", auc_value)


    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
    cm = confusion_matrix(Y_test, y_pred_binary)
    print (cm)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.show()

   
    plt.figure(1)
    plt.plot([0, 1], [0, 1], 'y--')
    plt.plot(fpr, tpr, marker='.')
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve')
    plt.show()


    plt.show()


```
### Step 8: Model Deployment

In this final step, the trained machine learning model is made accessible for practical use within the organization's systems. This often involves integrating the model into the company's existing software or platforms, allowing it to make real-time predictions on new customer data