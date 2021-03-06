
import string
import pickle
import time as t
import numpy as np
import pandas as pd
import missingno as msno
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import wordcloud as wc

from dask.diagnostics import ProgressBar
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, KFold, train_test_split, learning_curve
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import LabelEncoder, PolynomialFeatures
from sklearn.metrics import make_scorer, classification_report, confusion_matrix, accuracy_score, r2_score
from sklearn.metrics import roc_curve, roc_auc_score, average_precision_score, confusion_matrix, precision_recall_curve
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_selection import SelectKBest, chi2, f_classif, f_regression, mutual_info_classif, mutual_info_regression
from sklearn.linear_model import SGDRegressor

import nltk
from nltk.corpus import stopwords, wordnet
from nltk import pos_tag
from nltk.stem import WordNetLemmatizer


# DATA

def describe_data(df, info_flag=True):
    """
    Describe given dataframe
    """
    if info_flag:
        print(df)
        print('NaN:\n', df.isna().sum())
        print('NaN rows:\n', df[df.isna().any(axis=1)])
        print(df.describe(include='all'))
        msno.matrix(df)
        plt.show()
        plt.close()
    return

def get_wordnet_pos(pos_tag):
    """
    Returns the wordnet object value corresponding to the POS tag
    """
    #nltk.download('wordnet')
    if pos_tag.startswith('J'):
        return wordnet.ADJ
    elif pos_tag.startswith('V'):
        return wordnet.VERB
    elif pos_tag.startswith('N'):
        return wordnet.NOUN
    elif pos_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN

def clean_text(text, lemmatizer):
    """
    Clean given text
    """
    # To lower case
    text = text.lower()

    # Tokenization
    tokens = text.split(' ')

    # Remove punctuation
    words = [token.strip(string.punctuation) for token in tokens]

    # Remove words that contains numbers
    words = [word for word in words if not any(c.isdigit() for c in word)]

    # Remove words with less than 2 letters
    words = [word for word in words if len(word) > 1]

    # Remove stop words
    #nltk.download('stopwords')
    stop_words = stopwords.words('english')
    words = [word for word in words if word not in stop_words]

    # Part of Speach tagging
    #nltk.download('averaged_perceptron_tagger')
    pos_tags = pos_tag(words)

    # Lemmatize words - convert into root of the word
    words = [lemmatizer.lemmatize(word, get_wordnet_pos(tag)) for word, tag in pos_tags]

    # Remove words with less than 2 letters after lemmatization
    words = [word for word in words if len(word) > 1]

    return ' '.join(words)

def correlations(data):
    """
    Calculate correlation matrix
    """
    # Transform categorical into numeric feature
    encoder = LabelEncoder()
    data['Reviewer_Nationality_num'] = encoder.fit_transform(data['Reviewer_Nationality'])

    # Calculate correlation
    corr_matrix = data[['Reviewer_Nationality_num', 'Reviewer_Score']].corr(method='pearson')
    print(corr_matrix)

    # Show as heatmap
    plt.figure(figsize=(10, 10))
    sns.heatmap(corr_matrix, annot=True)
    plt.show()
    plt.close()
    
    return corr_matrix

def load_data(file_name, sample=False, info=False):
    """
    Load dataset
    """
    # Load data
    data = pd.read_csv(file_name, header=0)
    column_names = ['Negative_Review', 'Positive_Review', 'Reviewer_Nationality', 'Reviewer_Score']
    data = data[column_names]
    
    # Sample data
    if sample:
        data = data.sample(frac=0.1, replace=False, random_state=21)

    # Information
    if info:
        # Set style
        sns.set(style='whitegrid')

        # Describe
        describe_data(data, info_flag=info)

        # Class distribution
        positive = len(data[data['Positive_Review'] != 'No Positive'])
        negative = len(data[data['Negative_Review'] != 'No Negative'])
        sns.barplot(x=['Negative', 'Positive'], y=[negative, positive])
        plt.show()
        plt.close()  

        # Score distribution
        sns.violinplot(x=data['Reviewer_Score'])  
        plt.show()
        plt.close()

        # Correlation
        corr_matrix = correlations(data)

    return data

def load_data_predict_score(file_name, load_clean_data=True, sample=False, info=False):
    """
    Load data for regression task - predicting score from text review
    """
    # File path
    file_path = 'data/output/data_score.pkl'

    # Load clean data
    if load_clean_data:
        data = pd.read_pickle(file_path)
        X_data = data['Review']
        y_data = data['Reviewer_Score']
        return data, X_data, y_data

    # Load data
    data = load_data(file_name, info=info, sample=sample)
    data['Review'] = data['Negative_Review'] + data['Positive_Review']

    # Clean data
    lemmatizer = WordNetLemmatizer()
    data['Review'] = data['Review'].apply(lambda text: text.replace('No Negative', '').replace('No Positive', ''))
    data['Review'] = data['Review'].apply(lambda text: clean_text(text, lemmatizer))

    # Save clean data
    data.to_pickle(file_path)

    # Define input and output
    X_data = data['Review']
    y_data = data['Reviewer_Score']

    # Word cloud
    if info:
        show_word_cloud(X_data)

    return data, X_data, y_data

def load_data_sentiment_analysis(file_name, load_clean_data=True, sample=False, info=False):
    """
    Load data for classification task - sentiment analysis from text review
    """
    # File path
    file_path = 'data/output/data_sentiment.pkl'

    # Load clean data
    if load_clean_data:
        data = pd.read_pickle(file_path)
        X_data = data['Review']
        y_data = data['Sentiment']
        return data, X_data, y_data

    # Load data
    data = load_data(file_name, info=info, sample=sample)

    # Negative review
    negative = data[data['Negative_Review'] != 'No Negative']['Negative_Review']
    data_neg = pd.DataFrame([])
    data_neg['Review'] = negative
    data_neg['Sentiment'] = -1

    # Positive review
    positive = data[data['Positive_Review'] != 'No Positive']['Positive_Review']
    data_pos = pd.DataFrame([])
    data_pos['Review'] = positive
    data_pos['Sentiment'] = 1

    # Concant
    data = pd.concat([data_neg, data_pos], ignore_index=True)

    # Clean data
    lemmatizer = WordNetLemmatizer()
    data['Review'] = data['Review'].apply(lambda text: clean_text(text, lemmatizer))

    # Save clean data
    data.to_pickle(file_path)

    # Define input and output
    X_data = data['Review']
    y_data = data['Sentiment']

    # World Cloud
    if info:
        show_word_cloud(X_data)

    return data, X_data, y_data

def show_word_cloud(data):
    """
    Show the most frequnet words as cloud
    """
    # Create word cloud
    word_cloud = wc.WordCloud(
        background_color = 'white',
        max_words = 200,
        max_font_size = 40, 
        scale = 3
    ).generate(str(data))

    # Show word cloud
    plt.imshow(word_cloud)    
    plt.axis('off')
    plt.show()
    plt.close()
    return


# OPTIMIZATION

def optimization_random_search(model, X_train, y_train, is_regression=False):
    """
        About Random Search
            This has two main benefits over an exhaustive search (grid search):
            1. A budget can be chosen independent of the number of parameters and possible values.
            2. Adding parameters that do not influence the performance does not decrease efficiency.
            3. In grid search for one bad paramter value we continue to evaluate 10 bad models by calculating other combination of rest hparams,
                on the other hand random search always change value of hparamter, it's not fixed.

            For each parameter, either a distribution over possible values or a list of discrete choices (which will be sampled uniformly) can be specified.
            Additionally, a computation budget, being the number of sampled candidates or sampling iterations, is specified using the n_iter parameter.
            For continuous parameters, such as C above in SVM, it is important to specify a continuous distribution to take full advantage of the randomization
            Usually the result in parameter settings is quite similar, while the run time for randomized search is drastically lower and 
                the performance is may slightly worse for the randomized search.

            Source: https://scikit-learn.org/stable/auto_examples/model_selection/plot_grid_search_digits.html
            Pipeline: https://scikit-learn.org/stable/auto_examples/model_selection/grid_search_text_feature_extraction.html#sphx-glr-auto-examples-model-selection-grid-search-text-feature-extraction-py
            GridSearch: https://scikit-learn.org/stable/auto_examples/model_selection/plot_grid_search_digits.html

        About Pipeline
            Define workflow
                It sequentially apply a list of transforms and a final estimator 
                It fit/train transformers on training set and transforms training and test set instances before applying final estimator
                The transformers in the pipeline can be cached using memory argument.
            Source: https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html

        About TF-IDF Vectorizer
            min_df: When building the vocabulary ignore terms that have a document frequency strictly lower than the given threshold.
                This value is also called cut-off in the literature. 
                If float in range of [0.0, 1.0], the parameter represents a proportion of documents, integer absolute counts. 
                This parameter is ignored if vocabulary is not None.
            Source: https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html

        About Trunced SVD
            Dimensionality reduction using truncated SVD (aka LSA).
            This transformer performs linear dimensionality reduction by means of truncated singular value decomposition (SVD). 
            Contrary to PCA, this estimator does not center the data before computing the singular value decomposition. 
            This means it can work with sparse matrices efficiently.
            Source: https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.TruncatedSVD.html#sklearn.decomposition.TruncatedSVD

        About Feature Selection
            Select features according to the k highest scores by using given function
            Source: https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectKBest.html

        About Make Scorer function
            it makes a scorer from a performance metric or loss function, highest score (or smallest loss if specified)
            Source: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.make_scorer.html
    """

    # Define workflow
    pipeline = Pipeline([
        ('tf-idf', TfidfVectorizer()), 
        ('trunc-svd', TruncatedSVD()),
        ('select-k', SelectKBest()), 
        ('estimator', model)
    ])
    
    # Define Hyperparamter space
    score_funcs = [f_regression] if is_regression else [f_classif]
    criterions = ['mse'] if is_regression else ['entropy']

    hparams = [{
                'tf-idf__min_df': np.arange(3, 7),                 # ignore terms that have a document frequency strictly lower than the given threshold
                'trunc-svd__n_components': np.arange(50, 70),       # desired dimensionality of output data
                'trunc-svd__n_iter': np.arange(3, 7),              # number of iterations for randomized SVD solver
                'select-k__score_func': score_funcs,                    # scoring function for feature selection
                'select-k__k': np.arange(25, 50),                       # number of feature to select with best score
                'estimator__n_estimators': np.arange(100, 250),          # the number of trees in the forest
                'estimator__criterion': criterions,                     # the function to measure the quality of a split
                'estimator__max_depth': np.arange(50, 150),               # the maximum depth of the tree
                'estimator__min_samples_split': np.arange(15, 550),      # the minimum number of samples required to split an internal node
                'estimator__min_samples_leaf': np.arange(15, 550),        # the minimum number of samples required to be at a leaf node
                'estimator__max_features': ['sqrt', 'log2'],      # the number of features to consider when looking for the best split
                'estimator__bootstrap': [True],                  # whether bootstrap samples are used when building trees, if False, the whole dataset is used to build each tree
                'estimator__max_samples': stats.uniform(0.7, 0.25),        # if bootstrap is True, the number of samples to draw from X to train each base estimator
                'estimator__n_jobs': [-1]                               # the number of jobs to run in parallel
    }]

    # Define metrics
    #   For each trained model with specific combination of hparam run this metrics (in each iteration of CV)
    #   GridSearchCV.cv_results_ will return scoring metrics for each of the score types provided
    if is_regression: 
        metrics = {'r2' : make_scorer(r2_score)}
    else:
        metrics = {'accuracy' : make_scorer(accuracy_score)}
        
    # Define best estimator metric
    #   Refit an estimator on the whole dataset (full training data) using the best found parameters
    #   For multiple metric evaluation, this needs to be a str denoting the scorer/metric that would be used to find the best parameters for refitting the estimator at the end.
    if is_regression: 
        refit_metric = 'r2'
    else:
        refit_metric = 'accuracy'

    # Definition of search strategy
    if is_regression:
        cross_validation = KFold(n_splits=3, shuffle=True, random_state=0) 
    else: 
        cross_validation = StratifiedKFold(n_splits=3, shuffle=True, random_state=0) 
    
    print('Tuning hyperparameters for metrics=', metrics.keys())
    rand_search = RandomizedSearchCV(
        estimator=pipeline, 
        param_distributions=hparams, 
        scoring=metrics,
        n_iter=3,
        refit=refit_metric,
        cv=cross_validation,                                            # for every hparam combination it will shuffle data with same key, random_state
        return_train_score=True,                                        # used to get insights on how different parameter settings impact the overfitting/underfitting trade-off
        n_jobs=-1,                                                      # number of jobs to run in parallel, None means 1 and -1 means using all processors
        verbose=10)                                                     

    # Searching
    print('Performing Random search...')
    print('Pipeline:', [name for name, _ in pipeline.steps])
    start_time = t.time()
    with ProgressBar():
        rand_search.fit(X_train, y_train)                              # find best hparameters using CV on training dataset
    print('Done in {:.3f}\n'.format((t.time() - start_time)))
    
    # Results
    print('Cross-validation results:')
    results = pd.DataFrame(rand_search.cv_results_)
    columns = [col for col in results if col.startswith('mean') or col.startswith('std')]
    columns = [col for col in columns if 'time' not in col]
    results = results.sort_values(by='mean_test_'+refit_metric, ascending=False)
    results = results[columns].round(3).head()
    results.reset_index(drop=True, inplace=True)
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):            # more options can be specified also
        print(results)

    # Best score
    print('\nBest score: {:.3f}'.format(rand_search.best_score_))         # mean cross-validated score of the best_estimator (best hparam combination)

    # Best estimator
    best_estimator = rand_search.best_estimator_                          # estimator that was chosen by the search, i.e. estimator which gave highest score

    # Hparams of best estimator
    print('\nBest parameters set:')
    best_parameters = best_estimator.get_params()
    for param_name in sorted(hparams[0].keys()):
        print('\t{}: {}'.format(param_name, best_parameters[param_name]))
            
    # Save model
    model_name = 'model_score.pkl' if is_regression else 'model_sentiment.pkl'
    with open('data/output/' + model_name, 'wb') as file:
        pickle.dump(rand_search, file)

    # Visualize the Pipeline process
    visualize_pipeline(pipeline, 'estimator_by_grid')

    return best_estimator

def visualize_pipeline(pipeline, file_name):
    """
    Visualize the Pipeline process contained byall transformations and final estimator
    Input
        Instance of Pipeline class
    Output
        HTML file with visualization of pipeline process
    Source: https://scikit-learn.org/stable/modules/compose.html#feature-union
    """
    from sklearn.utils import estimator_html_repr
    with open('data/output/{}.html'.format(file_name), 'w') as f:
        f.write(estimator_html_repr(pipeline))
    return


# PREDICT SCORE

def predict_score(load_model=False, load_clean_data=True, sample=False, info=True):
    """
    Predict score from text review
    """
    # Load and split data
    data, X_data, y_data = load_data_predict_score('data/reviews/Hotel_Reviews.csv', load_clean_data=load_clean_data, sample=sample, info=info)
    X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.1, random_state=0)

    # Load saved model
    if load_model:
        with open('data/output/model_score.pkl', 'rb') as file:
            estimator = pickle.load(file)
    else:
        # Create new features using non-linear method 'polynomial'
        poly_features = PolynomialFeatures(degree=3)

        # Model
        model = SGDRegressor()

        estimator = make_pipeline(TfidfVectorizer(min_df=20), SelectKBest(k=300), poly_features, model)
        estimator.fit(X_train, y_train)

        # Save model
        model_name = 'model_score.pkl'
        with open('data/output/' + model_name, 'wb') as file:
            pickle.dump(estimator, file)

        # Visualize the Pipeline process
        visualize_pipeline(estimator, 'estimator_score')

        # Find best estimator
        #model = RandomForestRegressor(n_jobs=-1)
        #estimator = optimization_random_search(model, X_train, y_train, is_regression=True)

    # Prediction
    scores_train = estimator.score(X_train, y_train)
    print('Random forest regressor on train dataset:\t {:.4f}'.format(scores_train))

    scores_test = estimator.score(X_test, y_test)
    print('Random forest regressor on test dataset:\t {:.4f}\n'.format(scores_test))

    # Visualization
    #plot_learning(estimator, X_train, y_train, n_iter=5, is_regression=True)

    return

def plot_learning(model, X_train, y_train, n_iter=5, is_regression=False):
    """
    A learning curve shows the validation and training score of an estimator for varying numbers of training samples.
    It is a tool to find out how much we benefit from adding more training data 
        and whether the estimator suffers more from a variance error or a bias error.
    fit_time
        The time for fitting the estimator on the train set for each cv split.

    Source: https://scikit-learn.org/stable/modules/learning_curve.html
    """
    metric = 'accuracy' if not is_regression else 'r2_score'
    train_sizes, train_scores, val_scores, fit_times, _ = learning_curve(
        model, 
        X_train, 
        y_train, 
        train_sizes=np.linspace(0.1, 1.0, n_iter), 
        scoring=metric, 
        cv=3,
        n_jobs=-1,
        return_times=True)
    
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)

    val_scores_mean = np.mean(val_scores, axis=1)
    val_scores_std = np.std(val_scores, axis=1)

    fit_times_mean = np.mean(fit_times, axis=1)
    fit_times_std = np.std(fit_times, axis=1)

    # Plot learning curve
    plt.figure()
    plt.title('Learning Curve')
    plt.xlabel('Training examples')
    plt.ylabel(metric)
    plt.ylim(0.0, 1.1)

    plt.plot(train_sizes, train_scores_mean, 'o-', color='r', label='Training score')
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, color='r', alpha=0.1)

    plt.plot(train_sizes, val_scores_mean, 'o-', color='g', label='Cross-validation score')
    plt.fill_between(train_sizes, val_scores_mean - val_scores_std, val_scores_mean + val_scores_std, color='g', alpha=0.1)
    plt.legend(loc='best')

    # Plot n_samples vs fit_times
    plt.figure()
    plt.title('Scalability of the model')
    plt.xlabel('Training examples')
    plt.ylabel('Fit times')
    plt.plot(train_sizes, fit_times_mean, 'o-')
    plt.fill_between(train_sizes, fit_times_mean - fit_times_std, fit_times_mean + fit_times_std, alpha=0.1)

    # Plot fit_time vs score
    plt.figure()
    plt.title('Performance of the model')
    plt.xlabel('Fit times')
    plt.ylabel(metric)
    plt.plot(fit_times_mean, val_scores_mean, 'o-')
    plt.fill_between(fit_times_mean, val_scores_mean - val_scores_std, val_scores_mean + val_scores_std, alpha=0.1)

    plt.show()
    plt.close()
    return


# SENTIMENT ANALYSIS

def sentiment_analysis(load_model=False, load_clean_data=True, sample=False, info=True):
    """
    Predict sentiment from text review
    """
    # Load and split data
    data, X_data, y_data = load_data_sentiment_analysis('data/reviews/Hotel_Reviews.csv', load_clean_data=load_clean_data, sample=sample, info=info)
    X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.1, stratify=y_data, random_state=0)

    # Load saved model
    if load_model:
        with open('data/output/model_sentiment.pkl', 'rb') as file:
            estimator = pickle.load(file)
    else:
        # Find best estimator
        model = RandomForestClassifier(n_jobs=-1)
        estimator = optimization_random_search(model, X_train, y_train, is_regression=False)

    # Prediction
    scores_train = estimator.score(X_train, y_train)
    print('Random forest classifier on train dataset:\t {:.4f}'.format(scores_train))

    scores_test = estimator.score(X_test, y_test)
    print('Random forest classifier on test dataset:\t {:.4f}\n'.format(scores_test))
    
    y_preds = estimator.predict(X_test)
    print(classification_report(y_test, y_preds), '\n')

    # Visualization
    visualization_classification(y_test, y_preds)

    #plot_learning(estimator, X_train, y_train, n_iter=5, is_regression=False)

    return

def plot_roc_curve(y_test, y_scores, label='scores'):
    """
    ROC CURVE 
        It will also change the number of FPR and TRP by changing the treshold (all similar like precision-recall curve)
        One treshold value correspond to one FPR and TRP value, except some treshold can be removed because they are not relevant for plotting the curve
        Source: https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html
    """
    fpr, tpr, thresholds = roc_curve(y_test, y_scores)
    rocauc_score = roc_auc_score(y_test, y_scores)
    # Default Threshold
    temp = 0.5 if label == 'probabilities' else 0
    def_threshold_ind = np.argmin(np.abs(thresholds - temp))     # return index of aprox. default treshold value (for probability and score)
    fpr_def_threshold = fpr[def_threshold_ind]
    tpr_def_threshold = tpr[def_threshold_ind]
    # Plot
    plt.figure()
    plt.title('ROC curve', fontsize=16)
    plt.xlabel('False Positve Rate')
    plt.ylabel('True Positive Rate')
    plt.plot(fpr, tpr, lw=3, label='ROC curve with AUC score={:.2f}'.format(rocauc_score))
    plt.plot([0,1], [0,1], lw=3, color='navy', linestyle='--', label='ROC of Random classifier', alpha=0.7)
    plt.plot(fpr_def_threshold, tpr_def_threshold, marker='o', markersize=10, fillstyle='none', c='r', mew=3, alpha=0.8)
    plt.legend(loc='lower right')
    #plt.axes().set_aspect('equal')
    plt.show()
    plt.close()
    return

def plot_precision_recall_curve(y_test, y_scores, label='scores'):
    """
    PRECISION-RECALL CURVE
        Tresholds are scores, for each treshold(selected score) we can calculate precision and recall value
        We can plot all these scores thresholds values with Precision-Recall curve to observe precision and recall relationship for specific treshold.
        Source: https://scikit-learn.org/stable/auto_examples/model_selection/plot_precision_recall.html
    """
    precision, recall, thresholds = precision_recall_curve(y_test, y_scores)
    avg_precision = average_precision_score(y_test, y_scores)   # Average precision - summerize a curve
    # Default Threshold
    temp = 0.5 if label == 'probabilities' else 0               # by setting temp value you can manupilate with treshold and call make_predictions function
    def_threshold_ind = np.argmin(np.abs(thresholds - temp))    # return index of aprox. default treshold value (for probability and score)
    precision_def_threshold = precision[def_threshold_ind]
    recall_def_threshold = recall[def_threshold_ind]
    # Plot
    plt.figure()
    plt.title('Precision-Recall chart [{}]'.format(label), fontsize=16)
    plt.xlabel('Precision')
    plt.ylabel('Recall')
    plt.plot(precision, recall, lw=3, label='PR Curve with AUC score or AP={:.2f}'.format(avg_precision))
    plt.plot(precision_def_threshold, recall_def_threshold, marker='o', markersize=10, fillstyle='none', c='r', mew=3, alpha=0.8)
    plt.legend(loc='lower right')
    #plt.axes().set_aspect('equal')
    plt.show()
    plt.close()
    return

def plot_confusion_matrix(y_test, y_scores):
    """
    Plot confusion matrix as heatmap
    """
    index = ['Negative', 'Positive']
    conf_matrix = confusion_matrix(y_test, y_scores)
    df = pd.DataFrame(conf_matrix, index=index, columns=index)
    plt.figure(figsize = (5.5, 4))
    axes = sns.heatmap(df, annot=True, fmt='d', cmap='Blues')
    axes.set_ylabel('Actual', fontsize=14)
    axes.set_xlabel('Predicted', fontsize=14)
    plt.show()
    plt.close()
    return

def visualization_classification(y_test, y_preds):
    """
    Visualization of results of classification task 
    """
    plot_roc_curve(y_test, y_preds)

    plot_precision_recall_curve(y_test, y_preds)

    plot_confusion_matrix(y_test, y_preds)
    return
    

if __name__ == "__main__":

    load_model = True

    predict_score(load_model=load_model, load_clean_data=True, sample=False, info=False)

    sentiment_analysis(load_model=load_model, load_clean_data=True, sample=False, info=False)
    

