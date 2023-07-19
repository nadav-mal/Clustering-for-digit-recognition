# -*- coding: utf-8 -*-
"""
Created on Sat Mar  4 11:52:37 2023

Submitted by:
    Nadav Malul 206762973
    Ely Asaf 319027355
"""
#%%
import numpy as np
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
import matplotlib.pyplot as plt
import random
from sklearn.decomposition import PCA
from time import time #Used to calculate time each idea takes to compute
from sklearn.ensemble import VotingClassifier
from sklearn.neighbors import KNeighborsClassifier

#%%

def get_dataset():
    
    # Load data from https://www.openml.org/d/554
    print('fetching data from openml')
    X, y = fetch_openml(
        "mnist_784", version=1, return_X_y=True, as_frame=False, parser="pandas"
    )
    X = X / 255.0 #normalize data to 0-1
    # Split data into train partition and test partition
    return train_test_split(X, y, random_state=0, test_size=0.3)
#%%

def print_dataset_shape(x_train, x_test, y_train, y_test):
    print('Training Data: {}'.format(x_train.shape))
    print('Training Labels: {}'.format(y_train.shape))
    print('Testing Data: {}'.format(x_test.shape))
  
#%%
'''This function is used to choose between a best KMeans clustering 
configuration by multithreading and Bench comparing between multiple
configurations. note, you will get an alert if NUM_OF_THREADS
isn't defined as an Environment variable on the local system'''
def bench_k_means(kmeans, name, data, labels):
    from sklearn import metrics
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    t0 = time()
    estimator = make_pipeline(StandardScaler(), kmeans).fit(data)
    fit_time = time() - t0
    results = [name, fit_time, estimator[-1].inertia_]

    # Define the metrics which require only the true labels and estimator
    # labels
    clustering_metrics = [
        metrics.homogeneity_score,
        metrics.completeness_score,
        metrics.v_measure_score,
        metrics.adjusted_rand_score,
        metrics.adjusted_mutual_info_score,
    ]
    results += [m(labels, estimator[-1].labels_) for m in clustering_metrics]

    # The silhouette score requires the full dataset
    results += [
        metrics.silhouette_score(
            data,
            estimator[-1].labels_,
            metric="euclidean",
            sample_size=300,
        )
    ]
    end_time = time()
    taken_time = end_time - t0
    print('Extra time taken because of the use of bench_k_means:')
    print(taken_time)
    return kmeans
#%%
'''preprocess_data asks the user to choose a preprocessing method for 
the input data. The options are to not preprocess, use PCA,
 normalize, or use minmax scaling. It takes in the inputs x_train and x_test 
and returns the preprocessed x_train and x_test data. The function also prints
 the chosen preprocessing method and the time taken to preprocess the data.'''
def preprocess_data(x_train, x_test):
        print('Preprocess data?\t options are:')
        print(' 1: Dont preprocess\n 2: use PCA\n 3: Normalize\n 4: Use minmax')
        decision = input()
        print('You chose:', end = ' ')
        t0 = time()
        if decision == '1':
            print('dont preprocess')
            
        elif decision == '2':
            print('preprocess using PCA')
            x_train, x_test = preprocess_pca(x_train, x_test)
        elif decision == '3':
            print('Preprocess using normalize')
            x_train, x_test = normalize(x_train, x_test)
        elif decision == '4':
            print('Use minmax preprocess')
            x_train , x_test = minmax_preprocessing(x_train, x_test)
        else:
            print('dont preprocess.')
        t_end = time()
        t_total = t_end-t0
        print('Time taken to preprocess data:', end=' ')
        print(t_total)
        return x_train, x_test
'''Preprocessing train,test data using minmax'''
def minmax_preprocessing(X_train, X_test):
    from sklearn.preprocessing import MinMaxScaler
    # Create a MinMaxScaler object and fit it to the training data
    scaler = MinMaxScaler()
    scaler.fit(X_train)
    # Apply the scaler to the training and test data
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled
'''Preprocessing train,test data using PCA. find_max_pixels will decide
the number of features for n_componenets (new feature dimention)'''
def preprocess_pca(x_train, x_test):
    #returns the number of max pixels from the image with most pixels in the set
    def find_max_pixels(x_train):
        max_pixels = 0
        for x in x_train:
            curr_pixels_sum = sum(1 for pixel in x if pixel > 0)
            max_pixels = max(max_pixels, curr_pixels_sum)
        return max_pixels
    #784 is the current number of features for each image
    wanted_n = min(784,find_max_pixels(x_train))
    
    x_train = PCA(n_components=wanted_n).fit_transform(x_train)
    x_test = PCA(n_components=wanted_n).fit_transform(x_test)
    return x_train, x_test
def normalize(x_train, x_test):
    x_train = x_train / 255.0
    x_test = x_test / 255.0
    return x_train , x_test

#%%
'''get_classifier_from_user prompts the user to choose between three 
classifiers: SVM, KNN, or a combination of SVM and KNN using 
a voting classifier. It returns the selected classifier as an object. 
If the user inputs an invalid choice, it will prompt the user to enter a valid choice.'''
def get_classifier_from_user():
    while True:
        choice = input("Enter 1 for SVM, 2 for KNN, or 3 for a voting classifier on KNN and SVM:\n")
        if choice == '1':
            print('You chose: SVM')
            return LinearSVC(random_state=0, max_iter=1000, dual=False)
        elif choice == '2':
            print('You chose: KNN')
            return KNeighborsClassifier(n_neighbors=5)
        elif choice == '3':
            print('You chose: combination between svm,knn using voting')
            svm = LinearSVC(random_state=0, max_iter=1000, dual=False)
            knn = KNeighborsClassifier(n_neighbors=5)
            return VotingClassifier(estimators=[('svm', svm), ('knn', knn)], voting='hard')
        else:
            print("Invalid choice. Please enter 1, 2, or 3.")

#%%
'''This function takes in a dataset X_train and prompts the user to choose
 between KMeans and MiniBatchKMeans clustering algorithms. It then fits the 
 chosen algorithm to the training data and returns the fitted model.'''
def fit_kmeans(X_train):
    choise = input("Insert 1 for KMeans. 0 for Minibatch\n")
    if choise == '1':
        kmeans = KMeans(n_clusters=10, random_state=0, n_init=10)
    else:
        kmeans = MiniBatchKMeans(n_clusters=10, random_state=0)

    # Fit the KMeans model to the training data
    kmeans.fit(X_train)
    
    return kmeans
'''Inputs:
X: array-like, shape (n_samples, n_features). Input data
y: array-like, shape (n_samples,). The target values (class labels in classification, real numbers in regression).
optimize_kmeans: bool, whether to use optimized kmeans or minibatch kmeans
verbose: bool, whether to print additional information during the function's execution
Output:
classifiers: list of fitted classifiers for each cluster
kmeans: KMeans or MiniBatchKMeans object, fitted to the input data
Functionality:
Performs KMeans or MiniBatchKMeans clustering on the input data to split them into 10 clusters
Allows the user to choose a classifier for the task
Trains a separate classifier for each cluster using the chosen classifier
Returns a list of fitted classifiers and the KMeans or MiniBatchKMeans object used for clustering.'''
def get_classifiers_and_clusters(X,y, optimize_kmeans, verbose=False):
    from sklearn.base import clone

    classifiers = []
    kmeans = None
    if optimize_kmeans:
        kmeans = KMeans(init="k-means++", n_clusters=10, n_init=4, random_state=0)
        bench_k_means(kmeans=kmeans, name="k-means++", data=X, labels=y)
    else:
        kmeans = fit_kmeans(X)
    n_digits = 10
    
    # Prompt user to choose a classifier before the loop
    chosen_classifier = get_classifier_from_user()
    for i in range(n_digits):
        X_cluster = X[kmeans.labels_ == i] #All elements of the i'th element
        y_cluster = y[kmeans.labels_ == i] #Ground truth(labels) of those elements
        # Split the data into training and testing sets. test size is very small because its not the real test
        X_train, X_test, y_train, y_test = train_test_split(X_cluster, y_cluster, test_size=0.3, random_state=0)
        
        #Only visible by manualy toggeling the function argument to True
        if verbose:
            print('-' * 50)
            print_dataset_shape(X_train, X_test, y_train, y_test)
            print('-' * 50)
        # Training a classifier         
        classifier = clone(chosen_classifier)
        classifier.fit(X_train, y_train)
        classifiers.append(classifier)
    
    return classifiers, kmeans



#%%
'''
    Params:
        X: training data
        y: training labels
    Returns a single classifier(chosen by the user) which learnt on each cluster seperatly.
    used for Manual runs on dataset only
'''
def get_classifier_and_clusters(X,y):
    chosen_classifier = get_classifier_from_user()
    kmeans = fit_kmeans(X)
    return chosen_classifier, kmeans  
#%%


#%%
'''
Params:
    X_test, y_test: training set and labels.
    kmeans: a kmeans structure already fitted to x_test
    classifiers: an array of 10 classifiers, each classifier is trained on a 
    different cluster. (overall there are 10 clusters)
    
returns:
    the prediction for this method:
    for each image in y_test first we check to which cluster it should be assinged
    based on kmeans decision, afterwards we use the classifier which already trained
    on this cluster's data (using the ground truth labels of the data inside)
'''
    #-------------------
def evaluate_classifiers_accuracy(X_test, y_test, kmeans, classifiers, classificationMethod):
    y_pred = np.zeros_like(y_test)
    cluster_labels = kmeans.predict(X_test)
    
    for i, classifier in enumerate(classifiers):
        for j in range(kmeans.n_clusters):
            cluster_indices = np.where(cluster_labels == j)[0]
            X_cluster = X_test[cluster_indices]
            y_cluster = y_test[cluster_indices]
            
            if len(X_cluster) > 0:
                classifier.fit(X_cluster, y_cluster)
                y_cluster_pred = classifier.predict(X_cluster)
                y_pred[cluster_indices] = y_cluster_pred
    
    return y_pred

	
#%%

#%%
'''
Params:
    x_train,x_test,y_train,y_test: mnist784 digits toyset(after train test split) 
        
    benchK- a boolean which decides if user wants to use
    bench_kmeans algorithm to find 'optimal' configuration
    of k_means for the given trainset. (if false, default kmeans algo)
    
    title- title of kmeans to describe the proccess to the user
'''
def train_on_clustered_data(x_train, x_test, y_train, y_test,benchK, title):
  #We dont use this. we didnt remove because of bad mojo removing things the last second
    def evaluate_single_accuracy(X_test, y_test, kmeans, classifier):
        cluster_labels = kmeans.predict(X_test)
        X_test_clustered = np.column_stack((X_test, cluster_labels))
        y_pred = classifier.predict(X_test_clustered)
        accuracy = accuracy_score(y_test, y_pred)
        print('Accuracy: %.2f' % accuracy)

    
    #----Room for preprocessing here!!----
    x_train, x_test = preprocess_data(x_train, x_test)
    #---------------------------------------
    print(title)
    #Start timer
    t0 = time()
    classifiers, kmeans = get_classifiers_and_clusters(x_train, y_train,benchK)
    prediction = evaluate_classifiers_accuracy(x_test, y_test, kmeans, classifiers,title)
    #Stop timer
    t_end = time()
    end_time = t_end - t0
    #Print score
    print_classifier_score(y_test, prediction, end_time)
    
    #Plot pca model of clusters and their centroids among the x_train points and x_test. 
    #Feel free to remove those '#' and see the plot!
    #title1 = 'on train set. ' + title
    #title2 = 'on test set. ' + title
    #plot_kmeans_clusters(x_train, 10, kmeans, title1)
    #plot_kmeans_clusters(x_test, 10, kmeans, title2)

#%%
'''
Params:
    try_kbench: a boolean to choose if use bench_kmeans or regular kmeans
    configDescription: the description of the way we chose to initialize kmeans model

function first uses kmeans to cluster the dataset while also training an svm classifier
on each clustered data.
afterwards, function evaluates the performance of a model which randomly chooses a 
classifier from the classifiers array instead of choosing the model which trained on
the co-responding cluster. (results arn't good.)
'''
def run_third_idea(x_train, x_test, y_train, y_test,try_kbench, configDescription):
    '''
    evaluates y_test result compared to the results given from a random
    classifier
    '''
    def evaluate_randomly(X_test, y_test, kmeans, classifiers):
        y_pred = np.zeros_like(y_test)
        cluster_labels = kmeans.predict(X_test)
        for i in range(kmeans.n_clusters):
            mask = cluster_labels == i
            X_cluster = X_test[mask]
            # Choose a random classifier for this cluster.
            classifier = random.choice(classifiers)
            y_pred_cluster = classifier.predict(X_cluster)
            y_pred[mask] = y_pred_cluster
        return accuracy_score(y_test, y_pred)
    
    print(configDescription)
    
    # Room for preprocessing
    x_train, x_test = preprocess_data(x_train, x_test)
    #-----------------------
    classifiers, kmeans = get_classifiers_and_clusters(x_train, y_train,try_kbench)
    random_scores = []
    avg = 0
    for i in range(10):
        curr = evaluate_randomly(x_test, y_test, kmeans, classifiers)
        avg += curr
        random_scores.append(curr)
        
    print("average of 10 randomize:")
    avg = avg/10
    print(avg)
    #For testing remember to remove
    #print(get_distances_from_centroids(kmeans, x_test[0]))
    #print(y_test[0])
    
#%%
'''
Plots kmeans clusters in different colors using pca to reduce dimentions
'''
def plot_kmeans_clusters(X, n_clusters, kmeans, title):
    reduced_data = PCA(n_components=2).fit_transform(X)
    kmeans.fit(reduced_data)
    # Step size of the mesh. Decrease to increase the quality of the VQ.
    h = 0.02  # point in the mesh [x_min, x_max]x[y_min, y_max].
    # Plot the decision boundary. For that, we will assign a color to each
    x_min, x_max = reduced_data[:, 0].min() - 1, reduced_data[:, 0].max() + 1
    y_min, y_max = reduced_data[:, 1].min() - 1, reduced_data[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    # Obtain labels for each point in mesh. Use last trained model.
    Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.figure(1)
    plt.clf()
    plt.imshow(
        Z,
        interpolation="nearest",
        extent=(xx.min(), xx.max(), yy.min(), yy.max()),
        cmap=plt.cm.Paired,
        aspect="auto",
        origin="lower",
    )

    plt.plot(reduced_data[:, 0], reduced_data[:, 1], "k.", markersize=2, alpha = 0.2)
    # Plot the centroids as a white X
    centroids = kmeans.cluster_centers_
    plt.scatter(
        centroids[:, 0],
        centroids[:, 1],
        marker="x",
        s=169,
        linewidths=3,
        color="w",
        zorder=10,
    )
    plt.title(title)
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.xticks(())
    plt.yticks(())
    plt.show()
#%%

#%%
'''Printing results using scikit-learn metrics/ways to mesure classification quality
   Prints a confusion matrix, Accuracy score, F1 score and time taken.
 '''
def print_classifier_score(y_test, y_pred, end_time):
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    cm = confusion_matrix(y_test, y_pred)
    print('-' * 50)
    print(f"Confusion matrix:\n{cm}")
    print('-' * 50)
    print(f"Accuracy score: {acc}")
    print(f"F1 score: {f1}")
    print(f"Time taken: {end_time} seconds")

'''This function takes in training and testing data and performs classification
 without clustering. It first preprocesses the data based on user input,
 initializes a classifier based on user input, trains the classifier on the
 training data, tests it on the testing data, and prints the confusion matrix,
 accuracy score, F1 score, and time taken for the classification.'''
def check_without_clustering(x_train, x_test, y_train, y_test):
    #preprocess options    
    x_train, x_test = preprocess_data(x_train, x_test)
    #Start timer
    t0 = time()
    # Initialize a LinearSVC classifier
    classifier = get_classifier_from_user()
    # Train the classifier on the training data
    classifier.fit(x_train, y_train)
    # Test the classifier on the testing data
    y_pred = classifier.predict(x_test)
    end_time = time()
    print_classifier_score(y_test, y_pred,end_time -t0 )
#%%
'''
    Params:
        kmeans:  a kmeans struct already fitted to training set
        x:    an image  (28*28 matrix)
    returns the distances from the image to all centroids.
'''
def get_distances_from_centroids(kmeans, x):
    centroids = kmeans.cluster_centers_
    distances = [np.linalg.norm(x - centroid) for centroid in centroids]
    return distances


'''This function takes in the training and testing data, as well as a KMeans
 object. It calculates the distance from each data point to each of the KMeans
 centroids and adds these distances as features to the data. The updated data
 is returned as numpy arrays.'''
def add_distances_to_data(x_train, x_test, kmeans):
    x_train_new = []
    for x in x_train:
        distances = get_distances_from_centroids(kmeans, x.reshape(1, -1))
        x_train_new.append(np.concatenate([x.reshape(-1), distances]))
    
    x_test_new = []
    for x in x_test:
        distances = get_distances_from_centroids(kmeans, x.reshape(1, -1))
        x_test_new.append(np.concatenate([x.reshape(-1), distances]))
    
    return np.array(x_train_new), np.array(x_test_new)
#----------------------------------------------
    x_train_new = []
    for x in x_train:
        distances = get_distances_from_centroids(kmeans, x.reshape(1, -1))
        x_train_new.append(np.concatenate([x.reshape(-1), distances]))
    
    x_test_new = []
    for x in x_test:
        distances = get_distances_from_centroids(kmeans, x.reshape(1, -1))
        x_test_new.append(np.concatenate([x.reshape(-1), distances]))
    
    return np.array(x_train_new), np.array(x_test_new)
'''This function preprocesses the data, chooses a classifier and k-means 
algorithm, adds distances from centroids to the set, predicts test set, 
and prints the score.'''
def classify_using_distances(x_train, x_test, y_train, y_test):
    #Preprocessing data (if user wants to)
    x_train , x_test = preprocess_data(x_train, x_test)
    #Choosing a classifier and kmeans algorithm
    classifier, kmeans = get_classifier_and_clusters(x_train, y_train)
    
    #Adding distances from centroids to the set, predicting test set and printing score
    t0 = time()
    x_train , x_test = add_distances_to_data(x_train, x_test, kmeans)
    classifier.fit(x_train, y_train)
    y_pred = classifier.predict(x_test)
    end_time = time()
    print_classifier_score(y_test, y_pred,end_time -t0 )
    return
#%%

#%% 
'''Function run_menu displays a menu of available test ideas and prompts
 the user to choose one. It then executes the chosen test idea by calling
 the corresponding function. The function allows the user to stop the menu
 by choosing option 9.'''
def run_menu():
    
    def displayOptions():
        print("Please insert test idea. options are:")
        print(" 1: Plain classifier on the data without clusterting")
        print(" 2: Cluster data, afterwards- train classifiers, one for each cluster")
        print(" 3: Same as 2 but use random classifiers when labeling x_test")
        print(" 4: Classify using distances from centroids")
    def clustering_config():
        bench_kmeansConfig = 'Using bench Kmeans config.'
        defaultConfig = 'Using default kmeans.'        
        choise = input("Insert 1 for bench_kmeans or 0 for KMeans/Minibatch KMeans ")
        if choise == '1':
            print('Using bench_kmeans')
            return True, bench_kmeansConfig
        print('Using default kmeans')
        return False, defaultConfig
    
    x_train, x_test, y_train, y_test = get_dataset()
    
    while(True):
        displayOptions()
        choise = input("Insert idea, current available: 2,3,4. or insert 9 to stop \n")
        
        #---------------------------------
        if choise == '1':
            check_without_clustering(x_train, x_test, y_train, y_test)
        if choise == '2':
            #user_choise, clusterConfiguration = clustering_config()
            train_on_clustered_data(x_train, x_test, y_train, y_test,False, 'not using benchKMeans')      
            
        #---------------------------------
        elif choise == '3':
            run_third_idea(x_train, x_test, y_train, y_test,False,'not using benchKMeans')
        #---------------------------------
        elif choise == '4':
            classify_using_distances(x_train, x_test, y_train, y_test)
        
        elif choise == '9':
            return
        print('-' * 30)
    
#%%
#Best solution reached by inserting '2 4 1' to input buffer. 0.998 Accuracy as seen in project.DOCX
run_menu()

