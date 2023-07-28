# Machine Learning Portfolio

Repository consists of programs that demonstrate supervised and unsupervised learning's implementation. Problems ranging from linear regression to density based clustering are present in this repository.The repository is divided as Classification, Regression and Clustering based on the tasks that are carried out. In each of the folders various algorithms are used.


## Contents
+ [About](#intro) 
+ [Applications](#applications)
+ [Terms](#terms)
+ [Tasks](#task)
+ [Learning Types](#learning)
+ [Steps Involved](#steps)
+ [Libraries Used](#library)
+ [Datasets Description](#program)
+ [LICENSE](LICENSE)


<a id="intro"></a><h2>About</h2>

Machine learning is a subfield of artificial intelligence (AI) that focuses on the development of algorithms and models that allow computers to learn from data and make predictions or decisions without being explicitly programmed. The primary goal of machine learning is to enable computers to learn from experience and improve their performance over time. They are generally used for problems where we need to predict or estimate something from the data.

<a id="applications"></a><h2>Applications</h2>
There are a wide range of applications of deep learning few are mentioned below:

+ Email Spam Detection
+ Credit Card Fraud Detection
+ Medical Diagnosis
+ Recommender Systems
+ Sentiment Analysis
  
<a id="terms"></a><h2>Terms</h2>
To get started with the deep learning we need to have knowledege about several terms used.The terms and their meaning are described below:

| TERMS        | MEANING       
| ------------- |:-------------:|
**Feature** | An individual input variable used in the training of a machine learning model.
**Model** | A mathematical representation that maps input features to output predictions.
**Hyperparameters** | Parameters that are set before training a machine learning model and control the learning process, such as the learning rate or the number of hidden layers in a neural network.
**Feature Engineering** | The process of selecting, transforming, and creating relevant features to improve the performance of a machine learning model.
**Loss Function** | A function that measures the difference between predicted and actual values, used to guide the training process.
**Cross-Validation** : A technique used to assess the performance of a model by dividing the data into subsets for training and testing.
**Training Set** | The part of the data used to train a machine learning model.
**Test Set** | The part of the data used to evaluate the performance of a trained machine learning model on unseen data.
**Validation Set** | An independent dataset used to tune hyperparameters and prevent overfitting during model training.
**Overfitting** | A situation where a machine learning model performs well on the training data but poorly on unseen data.
**Underfitting** | A situation where a machine learning model fails to capture the underlying patterns in the data.
**Gradient Descent** | An optimization algorithm used to update the model's parameters and minimize the loss function.
**Ensemble Learning** | A method that combines multiple models to improve prediction accuracy and generalization.
**Bias** | Difference between the actual value and the value predicted by the model.
**Variance** | The amount by which the prediction changes upon changing the training set.
**Precision** | It is a value given by ratio of true positives to predicted positives.
**Recall** | It is a value given by ratio of true positives to actual positives.


<a id="task"></a><h2>Tasks</h2>

| TERMS        | MEANING       
| ------------- |:-------------:|
**Clustering** |  A technique used to group similar data points together based on their similarity.
**Classification** | A task in supervised learning where the model predicts a categorical label or class.
**Regression** | A task in supervised learning where the model predicts a continuous numerical value.


<a id="learning"></a><h2>Learning Type</h2>

| TERMS        | MEANING       
| ------------- |:-------------:|
**Supervised Learning** | A type of machine learning where the model is trained on labeled data, i.e., input-output pairs, and learns to make predictions on new, unseen data.
**Unsupervised Learning** | A type of machine learning where the model is trained on unlabeled data and learns patterns and structures from the data without explicit output labels.
**Semi-Supervised Learning** | A combination of supervised and unsupervised learning, where the model is trained on a combination of labeled and unlabeled data.
**Reinforcement Learning** | A type of machine learning where an agent learns to make decisions by interacting with an environment and receiving feedback in the form of rewards or penalties.

<a id="steps"></a><h2> Steps Involved</h2>
| Step        | Task       
| ------------- |:-------------:|
**Data Collection** | The first step is to gather relevant data for the problem you want to solve. Data can come from various sources, such as databases, APIs, or manual data entry.
**Data Preprocessing** | Once you have collected the data, it needs to be cleaned and prepared for analysis. This step involves handling missing values, dealing with outliers, and converting data into a suitable format for machine learning algorithms.
**Feature Engineering** | Feature engineering is the process of selecting, transforming, or creating new features from the existing data to improve the performance of machine learning models.
**Data Splitting** |  Before training a machine learning model, the dataset is divided into training and testing sets. The training set is used to train the model, while the testing set is used to evaluate its performance on unseen data.
**Model Selection** | Depending on the problem type (e.g., classification or regression) and data characteristics, you choose an appropriate machine learning algorithm to train the model.
**Model Training** | During this step, the selected machine learning algorithm is applied to the training data to learn patterns and relationships in the data.
**Model Evaluation** | After training the model, it is evaluated on the testing data to assess its performance. Various metrics, such as accuracy, precision, recall, and mean squared error, are used to evaluate the model's performance.
**Hyperparameter Tuning** | Many machine learning algorithms have hyperparameters that control the learning process. Hyperparameter tuning involves searching for the best combination of hyperparameters to optimize the model's performance.
**Model Deployment** | Once you have a trained and tuned model, it can be deployed to make predictions on new, unseen data.
**Model Monitoring and Maintenance** | Machine learning models may require periodic monitoring and maintenance to ensure they continue to perform well as new data becomes available.
**Interpretation and Visualization** | Understanding how the model makes predictions is essential for building trust and gaining insights from the model's results. Interpretation and visualization techniques help explain the model's behavior and decision-making process.
**Iteration and Improvement** | Machine learning is an iterative process. After deploying the model, you may receive new data and feedback, leading to further improvements and updates to the model.


<a id="library"></a><h2> Libraries Used</h2>

+ **Numpy** : Used for numerical computations in python
+ **Pandas** : Used for file reading and other operations when working with large data.
+ **Sklearn** : This is a machine learning library for python.
+ **Matplotlib** : Visualization library
+ **Seaborn** : Interactive visualizations are made using these library.


<a id="program"></a><h2> Datasets Description</h2>

The datasets used for these program are downloaded from **kaggle**. Datasets can be found [here](https://github.com/Sandy0002/Machine-Learning-Exercises/tree/main/Datasets).

Since there are lot of programs so the datasets are used and tasks carried out using them are covered here.
| Datasets        | Description       
| ------------- |:-------------:|
**Coffee Dataset** | This dataset consists of information about coffee and its types. Here various tasks have been carried out in the programs such as finding the category of the coffee and amount of acidity in the coffee.
**Fuel Consumption** | Consists information about the vehicle,fuel,fuelconsumption etc. Programs such as estimating the class of the vehicle and fuel consumption are written.
**Garments Worker Productivity** |  The data consists of various columns consisting of dates, quarters,departments, productivity details etc. The programs to find out the productivity of the workers and department of the workers are there.
**Insurance** | This is a dataset that provides information about people and the insurance amount that are paid by them. The program estimating the premium to be paid by a person is there.
**Iphone Prices** | This dataset cosists data about Iphone such as its variant,lauch year,description and price of the variant. Program for estimating Iphone price is there.
**Loan Sanction** | This dataset consists of informtion about loan such as loan id, age of the person, gender, loan amount, loan status etc.Loan is used in various aspects.Hence programs for whether a person will get a loan or not and if gets then what will be the amount programs for such tasks are there in the repository.
**Possum Dataset** | This is an animal whose details are present in the dataset.Tasks such finding gender of the possum have been carried out.
**Telecom Dataset** | A very popular problem whether a person will churn or not. For determining status this dataset have been used.
**Titanic Dataset** | This dataset consists of 1912 famous tragedy the sinkage of Titanic data such as passenger id, class of the ticket, gender, survival status etc. Program for survival status is present in the repository.
**Tobacco Dataset** | This is dataset consists of information about the age of beginning of tobacco,bidi and cigratte of the children across various states of India. Tasks to estimate the average age of starting of tobacco and bidi have been carried out.
**Wine Quality** | This dataset contains data about red wine such as acidity,sugar content,chlorides, density,pH,quality etc. Programs for estimating the acidity and quality of the wine is present in the repository.

## LICENSE
[MIT LICENSE](LICENSE)
