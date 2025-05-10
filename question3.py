import numpy as np

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

# Import scikit-learn metrics module for accuracy and AUROC calculation
from sklearn import metrics


def conf_matrix(y_pred, y_true, num_class):
    """
    agrs:
    y_pred : List of predicted classes
    y_true : List of corresponding true class labels
    num_class : The number of distinct classes being predicted

    Returns:
    M : Confusion matrix as a numpy array with dimensions (num_class, num_class)
    """
    # Your code here. We ask that you not use an external library like sklearn to create the confusion matrix and code this function manually
    y_pred = np.array(y_pred, dtype=int)
    y_true = np.array(y_true, dtype=int)
    
    matrix = np.zeros((num_class, num_class), dtype=int)

    for true, pred in zip(y_true, y_pred):
        matrix[true, pred] += 1

    return matrix

def get_model(name, params):
    """
    args:
    name : Model name (string)
    params : list of parameters corresponding to given model

    Returns:
    model : sklearn model object
    """
    model = None
    if name == "KNN":
        k = params # Note that the expected parameters have already been extracted here
        # Define KNN model using sklearn KNeighborsClassifier object 
        # Note: you should include n_neighbors=k as an argument when initializing the model

        k = params
        model = KNeighborsClassifier(n_neighbors=k)

    elif name == "SVM":
        rand_state, prob = params # Note that the expected parameters have already been extracted here
        # Define SVM model using sklearn SVC object
        # Note: you should include random_state=rand_state and probability=prob as arguments when initializing the model
    
        rand_state, prob = params
        model = SVC(random_state=rand_state, probability=prob)

    elif name == "MLP":
        hl_sizes, rand_state, act_func = params # Note that the expected parameters have already been extracted here
        # Define MLP model using sklearn MLPClassifier object
        # Note: you should include hidden_layer_sizes=hl_sizes, random_state=rand_state, and activation=act_func when initializing the model

        model = MLPClassifier(hidden_layer_sizes=hl_sizes, random_state=rand_state, activation=act_func)

    else:
        print("ERROR: Model name not recognized/supported. Returned None")

    return model

def get_model_results(model_name, params, train_data, train_labels, test_data, test_labels, num_class):
    """
    args:
    model_name : Model name as a string
    params : List of parameters corresponding to the given model 
    train_Data : 5000x784 numpy array of FMNIST training images
    train_labels : corresponding 5000 numpy array of strings containing ground truth labels
    test_Data : 1000x784 numpy array of FMNIST test images
    test_labels : corresponding 1000 numpy array of strings containing ground truth labels
    num_class : integer number of unique classes being predicted

    Returns: 
    accuracy : Total model accuracy (numpy float) 
    confusion matrix: numpy array of dimensions (num_class,num_class)
    auc_score : Area under the curve of the ROC metric (numpy float)
    """
    # 1. Create Classifier model
    model = get_model(model_name, params)

    # 2. Train the model using the training sets 

    model.fit(train_data, train_labels)

    # 3. Predict the response for test dataset

    y_pred = model.predict(test_data)
   
    # 4. Model Accuracy, how often is the classifier correct? You may use metrics.accuracy_score(...)

    acc = metrics.accuracy_score(test_labels, y_pred)

    # 5. Calculate the confusion matrix by using the completed the function above 

    conf_mat = conf_matrix(y_pred, test_labels, num_class)

    # 6. Compute the AUROC score. You may use metrics.roc_auc_score(...)

    probabilities = model.predict_proba(test_data)
    auc_scor = metrics.roc_auc_score(test_labels, probabilities, multi_class='ovr')

    #acc, conf_mat, auc_scor = None, None, None # DELETE THIS LINE ONCE YOU HAVE CODED YOUR RESULTS 
    return acc, conf_mat, auc_scor

if __name__ == "__main__":

    # Step 1: Threshold quiz score into correct/incorrect
    dataset_1['correct'] = (dataset_1['s'] > 0.5).astype(int)

    # Step 2: Feature matrix and label vector
    features = ['fracSpent', 'fracComp', 'fracPaused', 'numPauses', 'avgPBR', 'numRWs', 'numFFs']
    X = dataset_1[features].to_numpy()
    y = dataset_1['correct'].to_numpy()

    # Step 3: Split into train/test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    num_class = 2  # Binary classification (0 = incorrect, 1 = correct)

    # KNN
    model_name = "KNN"
    for k in range(1,6):
        print(f"{k}-neighbors result:")
        params = k
        accuracy, confusion_matrix, auc_score = get_model_results(model_name, params, X_train, y_train, X_test, y_test, num_class)
        print("Accuracy:", accuracy)
        print("AUROC Score:", auc_score)
        print(confusion_matrix)
        print()
        
    # SVM
    model_name = "SVM"
    params = [1, True]
    accuracy, confusion_matrix, auc_score = get_model_results(model_name, params, X_train, y_train, X_test, y_test, num_class)
    print("SVM Result")
    print("Accuracy:", accuracy)
    print("AUROC Score:", auc_score)
    print(confusion_matrix)
    print()
    
    # MLP
    model_name = "MLP"
    params = [(15,10), 1, "relu"]
    accuracy, confusion_matrix, auc_score = get_model_results(model_name, params, X_train, y_train, X_test, y_test, num_class)
    print("MLP Result")
    print("Accuracy:", accuracy)
    print("AUROC Score:", auc_score)
    print(confusion_matrix)