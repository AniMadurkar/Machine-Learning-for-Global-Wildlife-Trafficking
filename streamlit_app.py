#Import statements
#C
from collections import Counter
import copy
#E
import en_core_web_sm
#I
from imblearn.over_sampling import SMOTE
#K
from kmodes.kprototypes import KPrototypes
#L
import lime
from lime import lime_tabular
#M
import matplotlib.pyplot as plt
#N
import numpy as np
#O
import os
#P
import pandas as pd
import pickle
#R
import re
import random
#S
import seaborn as sns
from scipy.stats.mstats import winsorize
from sklearn import cluster
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import learning_curve, ShuffleSplit, train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import label_binarize, LabelEncoder, MinMaxScaler, OneHotEncoder, RobustScaler, StandardScaler
from sklearn.tree import DecisionTreeClassifier
import streamlit as st
import streamlit.components.v1 as components
import string
import spacy
from spacy.lang.en import English
from spacy.lang.en.stop_words import STOP_WORDS
#W
from wordcloud import WordCloud
#X
import xgboost as xgb


def main():
    #Use the full page instead of a narrow central column
    st.set_page_config(layout="wide", initial_sidebar_state="expanded")

    #Main title on home page
    st.title("Machine Learning for Global Wildlife Trafficking")
    #Main option on top of sidebar
    st.sidebar.markdown("What would you like to do?")
    option = st.sidebar.radio("", ("Predict Action/Disposition from LEMIS", "Cluster Shipments from Panjiva"), index=0, key="option")
    
    #Set random seed
    RANDOM_SEED = 697

    @st.cache(persist=True, suppress_st_warning=True, allow_output_mutation=True)
    def loadData(dataset):
        """
        Input: 
            dataset = name of dataset
        Output:
            df = outputted dataset 
        """

        if dataset == 'LEMIS':
            #Read in data
            #Change filename below if the whole file is desired
            filename= 'lemis_cleaned_sample.csv'
            df = pd.read_csv(filename)

            #Creating one target column that has values for the disposition where the action was 'Refused' or the action was null
            df['action_disp'] = np.where((df['action'] == 'Refused') | (pd.isnull(df['action'])), df['disposition'], df['action'])
        elif dataset == 'Panjiva':
            #Read in data
            #Change filename below if the whole file is desired
            filename= 'panjiva_cleaned_sample.csv'
            df = pd.read_csv(filename)
            #Get a list of the columns with more than 800,000 nulls and drop them. Some human decision making used here. Keep the ones with a lot of nulls,
            #but drop the ones where nearly all rows have null. Probably should be based on a % into an auto tool. Although, with the miss_perc feature 
            #dropping columns probably isn't nessesary
            cols_drop = list(df.isna().sum()[df.isna().sum().sort_values() > 800000].sort_values()[::-1].index)
            df = df.drop(columns=cols_drop)

        return df

    if option == "Predict Action/Disposition from LEMIS":
        #Top title on the sidebar
        st.sidebar.title("Predicting Action/Disposition from LEMIS")

        @st.cache(persist=True, suppress_st_warning=True, allow_output_mutation=True)
        def featureEngineering(lemis):
            """
            Input:
                lemis = lemis dataframe
            Output:
                lemis_engineered = lemis dataframe with features engineered and appropriate columns
            """

            #Remove columns that are not needed and essentially can't use most of the units
            cols_drop = ['control_number', 'species_code', 'disposition_date', 'disposition_year', 'shipment_year',
                        'action', 'disposition', 'field', 'code', 'unit', 
                        'panjiva', 'centimeters', 'cubic centimeters', 'cubic meters', 'grams', 
                        'liters', 'meters', 'milligrams', 'milliliters', 'square centimeters', 
                        'square meters', 'unknown', 'quantity', 'weight (kg)', 'volume (teu)']
            lemis_engineered = lemis.drop(columns = cols_drop)

            taxonomy_vars = ['taxa', 'class', 'genus', 'species', 'subspecies', 'specific_name', 'generic_name']
            for c in taxonomy_vars:
                #Creating num field that yields 1 if taxonomy variable is filled out or not
                lemis_engineered[f'{c}_num'] = np.where(lemis_engineered[c].isna(), 0, 1)
                #Replacing nulls with blank text
                lemis_engineered[c] = lemis_engineered[c].replace(np.nan, '', regex=True)
            
            #Create numerical variables to make it easier to sum across
            taxonomy_numvars = [f'{t}_num' for t in taxonomy_vars]

            #Calculating percent complete for the taxonomy variables
            lemis_engineered['complete_percent'] = lemis_engineered[taxonomy_numvars].sum(axis=1) / 6
            #Concatenating all the taxonomy variables
            lemis_engineered['taxonomy_concat'] = lemis_engineered['taxa'] + ' ' + lemis_engineered['class'] + ' ' + lemis_engineered['genus'] \
                                                    + ' ' + lemis_engineered['species'] + ' ' + lemis_engineered['subspecies'] \
                                                        + lemis_engineered['specific_name'] + lemis_engineered['generic_name']
            lemis_engineered['taxonomy_concat'] = lemis_engineered['taxonomy_concat'].str.strip()

            #Add feature for if 'country_origin' and 'country_imp_exp' are the same or not
            lemis_engineered['country_origin_imp'] = np.where(lemis_engineered['country_origin'] == lemis_engineered['country_imp_exp'], 1, 0)
            #Creating label for if 'country_origin' is known or not and fixing nulls
            lemis_engineered['country_origin'] = np.where(lemis_engineered['country_origin'] == 'XX', 'unknown_origin', lemis_engineered['country_origin'])
            #Creating label for if any of these columns are known or not and fixing nulls
            for col in ['consignee', 'foreign_co', 'description', 'port', 'purpose', 'source']:
                lemis_engineered[col] = np.where(lemis_engineered[col].isna(), f'unknown_{col}', lemis_engineered[col])
                
            #Converting 'post_feb_2013' to int column
            lemis_engineered['post_feb_2013'] = np.where(lemis_engineered['post_feb_2013'].isna(), 0, 1)

            #Extracting the month of the year and the year from date column
            lemis_engineered['date'] = pd.to_datetime(lemis_engineered['date'])
            lemis_engineered['month'] = lemis_engineered['date'].dt.month.astype(str)

            #Replace any value_per_unit higher than the 95th percentile
            lemis_engineered['winsorized_value'] = winsorize(np.array(lemis_engineered['value']),limits = [0, 0.05])

            #Drop second set of unneeded columns
            cols_drop2 = taxonomy_vars + taxonomy_numvars + ['date', 'value', 'Unnamed: 0']
            lemis_engineered = lemis_engineered.drop(columns=cols_drop2)

            #Drop nan
            lemis_engineered = lemis_engineered.dropna()

            return lemis_engineered

        @st.cache(persist=True, suppress_st_warning=True, allow_output_mutation=True)
        def preprocessData(X_train, X_test, y_train, lemis_holdout_x):
            """
            Input:
                X_train = training vectors (unprocessed)
                X_test = testing vectors (unprocessed)
                y_train = labels for training vectors
                lemis_holdout_x = holdout set of rows
            Output:
                X_train_sm = scaled, one hot encoded, vectorized, and smoted training vectors
                X_train = scaled, one hot encoded, and vectorized training vectors
                X_test = transformed testing vectors on fitted preprocessing methods
                y_train_sm = smoted labels for training vectors
                y_train = labels for training vectors
                lemis_holdout_x = transformed holdout set of rows on fitted preprocessing methods
                columns_to_scale = list of numerical columns
                cat_one_hot_attribs = list of categorical columns
                txt_vect_attribs = list of text columns
            """

            #Define which columns should be encoded vs scaled vs vectorize
            columns_to_encode = ['country_origin', 'description', 'foreign_co', 'consignee', 
                                'port', 'purpose', 'source', 'month', 'country_imp_exp']
            columns_to_scale = ['winsorized_value', 'number of specimens', 'complete_percent']
            columns_to_vectorize = 'taxonomy_concat'

            #Creating all_features list for feature importances later
            all_features = columns_to_encode + columns_to_scale + [columns_to_vectorize] + ['post_feb_2013', 'country_origin_imp']

            #Creating pipeline to preprocess respective columns
            pipeline = ([('encode', OneHotEncoder(categories='auto', handle_unknown='ignore'), columns_to_encode),
                        ('scale', RobustScaler(), columns_to_scale),
                        ('vectorize', CountVectorizer(stop_words='english'), columns_to_vectorize)])
            #Sending pipeline through ColumnTransformer
            col_transform = ColumnTransformer(transformers=pipeline)

            #Fitting and transforming the training data, while only transforming the testing data
            X_train = col_transform.fit_transform(X_train)
            X_test = col_transform.transform(X_test)
            lemis_holdout_x = col_transform.transform(lemis_holdout_x)

            #Balance classes
            sm = SMOTE(random_state=RANDOM_SEED)
            X_train_sm, y_train_sm = sm.fit_resample(X_train, y_train)

            #Getting the encoders from the column transformer to get a list of attributes for categorical and text features
            cat_encoder = col_transform.transformers_[0][1]
            txt_encoder = col_transform.transformers_[2][1]
            cat_one_hot_attribs = list(cat_encoder.get_feature_names())
            x_col = [f"x{x}_" for x in list(range(0,11))]
            col_name = ['('+x+')' for x in columns_to_encode]
            for x,y in zip(col_name, x_col):
                cat_one_hot_attribs = list(map(lambda st: str.replace(st, y, x), cat_one_hot_attribs))
            txt_vect_attribs = list(txt_encoder.get_feature_names())

            return X_train_sm, X_train, X_test, y_train_sm, y_train, lemis_holdout_x, columns_to_scale, cat_one_hot_attribs, txt_vect_attribs
        
        @st.cache(persist=True, suppress_st_warning=True, allow_output_mutation=True)
        def dummyClassifier(X_train, X_test, y_train, y_test):
            """
            Input:
                X_train = scaled, one hot encoded, and vectorized training vectors
                X_test = transformed testing vectors on fitted preprocessing methods
                y_train = labels for training vectors
                y_test = labels for testing vectors
            Output:
                dummy_clf = dummy classifier fitted on training data
                y_pred_dummy = y predictions for dummy classifier
                y_test_binarize = labels for testing vectors binarized
                y_pred_dummy_binarize = y predictions for dummy classifier binarized
            """

            #Get dummy classifier predictions on regular traning set
            dummy_clf = DummyClassifier(random_state=RANDOM_SEED, strategy='most_frequent')
            dummy_clf = dummy_clf.fit(X_train, y_train)
            y_pred_dummy = dummy_clf.predict(X_test)

            #Binarizing y_test and y_pred for dummy classifier to make it easier for confusion matrix and other evaluation methods (due to multiclass)
            labels = [0,1,2,3]
            y_test_binarize = label_binarize(y_test, classes=labels)
            y_pred_dummy_binarize = label_binarize(y_pred_dummy, classes=labels)
            return dummy_clf, y_pred_dummy, y_test_binarize, y_pred_dummy_binarize

        @st.cache(persist=True, suppress_st_warning=True, allow_output_mutation=True)
        def predictionPipeline():
            """
            Input:
                None
            Output:
                X_train_sm = scaled, one hot encoded, vectorized, and smoted training vectors
                X_test = transformed testing vectors on fitted preprocessing methods
                y_train_sm = smoted labels for training vectors
                y_test = labels for testing vectors
                lemis_holdout_x = transformed holdout set of rows on fitted preprocessing methods
                lemis_holdout_y = labels for transformed holdout set of rows on fitted preprocessing methods
                columns_to_scale = list of numerical columns
                cat_one_hot_attribs = list of categorical columns
                txt_vect_attribs = list of text columns
                dummy_clf = dummy classifier fitted on training data
                y_pred_dummy = y predictions for dummy classifier
                y_test_binarize = labels for testing vectors binarized
                y_pred_dummy_binarize = y predictions for dummy classifier binarized
            """
            #Load in local file and conduct feature engineering
            lemis = loadData('LEMIS')
            lemis_engineered = featureEngineering(lemis)

            #Splitting the data into train and test sets
            lemis_train, lemis_test = train_test_split(lemis_engineered, test_size=0.2, shuffle=True, random_state=RANDOM_SEED)
            #Setting X,y,final test and then getting train and test arrays
            y = lemis_train.action_disp
            #Encode target labels with values ('Abandoned'=0, 'Cleared'=1, 'Reexport'=2, 'Seized'=3)
            le = LabelEncoder()
            y_enc = le.fit_transform(y)
            lemis_holdout_y = le.transform(lemis_test.action_disp)
            #Creating the instances that correspond to target labels
            X = lemis_train.drop(columns=['action_disp'])
            lemis_holdout_x = lemis_test.drop(columns=['action_disp'])
            #Splitting data into training and validation sets
            X_train, X_test, y_train, y_test = train_test_split(X, y_enc, test_size=0.2, random_state=RANDOM_SEED)

            #Preprocessing the data to scale, encode, vectorize, and smote. Also retrieving features after preprocessing
            X_train_sm, X_train, X_test, y_train_sm, y_train, lemis_holdout_x, \
                columns_to_scale, cat_one_hot_attribs, txt_vect_attribs = preprocessData(X_train, X_test, y_train, lemis_holdout_x)

            #Dummy classifier on processed training/validation sets for a baseline
            dummy_clf, y_pred_dummy, y_test_binarize, y_pred_dummy_binarize = dummyClassifier(X_train, X_test, y_train, y_test)

            return X_train_sm, X_test, y_train_sm, y_test, lemis_holdout_x, lemis_holdout_y, \
                    columns_to_scale, cat_one_hot_attribs, txt_vect_attribs, \
                        dummy_clf, y_pred_dummy, y_test_binarize, y_pred_dummy_binarize

        def constructLearningCurve(estimator, title, X, y, axes=None, ylim=None, cv=None,
                                    n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
            
            if axes is None:
                fig, axes = plt.subplots(1, 3, figsize=(20, 5))

            axes[0].set_title(title)
            if ylim is not None:
                axes[0].set_ylim(*ylim)
            axes[0].set_xlabel("Training examples")
            axes[0].set_ylabel("Score")

            train_sizes, train_scores, test_scores, fit_times, _ = \
                learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs,
                            train_sizes=train_sizes,
                            return_times=True)
            train_scores_mean = np.mean(train_scores, axis=1)
            train_scores_std = np.std(train_scores, axis=1)
            test_scores_mean = np.mean(test_scores, axis=1)
            test_scores_std = np.std(test_scores, axis=1)
            fit_times_mean = np.mean(fit_times, axis=1)
            fit_times_std = np.std(fit_times, axis=1)

            # Plot learning curve
            axes[0].grid()
            axes[0].fill_between(train_sizes, train_scores_mean - train_scores_std,
                                train_scores_mean + train_scores_std, alpha=0.1,
                                color="r")
            axes[0].fill_between(train_sizes, test_scores_mean - test_scores_std,
                                test_scores_mean + test_scores_std, alpha=0.1,
                                color="g")
            axes[0].plot(train_sizes, train_scores_mean, 'o-', color="r",
                        label="Training score")
            axes[0].plot(train_sizes, test_scores_mean, 'o-', color="g",
                        label="Cross-validation score")
            axes[0].legend(loc="best")

            # Plot n_samples vs fit_times
            axes[1].grid()
            axes[1].plot(train_sizes, fit_times_mean, 'o-')
            axes[1].fill_between(train_sizes, fit_times_mean - fit_times_std,
                                fit_times_mean + fit_times_std, alpha=0.1)
            axes[1].set_xlabel("Training examples")
            axes[1].set_ylabel("fit_times")
            axes[1].set_title("Scalability of the model")

            # Plot fit_time vs score
            axes[2].grid()
            axes[2].plot(fit_times_mean, test_scores_mean, 'o-')
            axes[2].fill_between(fit_times_mean, test_scores_mean - test_scores_std,
                                test_scores_mean + test_scores_std, alpha=0.1)
            axes[2].set_xlabel("fit_times")
            axes[2].set_ylabel("Score")
            axes[2].set_title("Performance of the model")

            return fig
        
        def plotLearningCurves(model, X_train_subset, y_train_subset):
            
            title = "Learning Curves"
            cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=RANDOM_SEED)
            fig = constructLearningCurve(model, title, X_train_subset, y_train_subset, cv=cv, n_jobs=4)
            st.pyplot(fig)

        def modelEvaluationMetrics(model, y_test, y_test_binarize, y_pred, y_pred_binarize, X_test):

            st.write('Accuracy Score : ', accuracy_score(y_test, y_pred).round(4))
            #Precision: What proportion of positive identifications was actually correct?
            st.write('Precision Score : ', precision_score(y_test, y_pred, average='weighted', labels=np.unique(y_pred)).round(4))
            #Recall: What proportion of actual positives was identified correctly?
            st.write('Recall Score : ' , recall_score(y_test, y_pred, average='weighted', labels=np.unique(y_pred)).round(2))
            #F1 Score: Weighted average of precision and recall
            st.write('F1 Score : ' , f1_score(y_test, y_pred, average='weighted', labels=np.unique(y_pred)).round(2))
            #ROC_AUC Score: Area Under the Receiver Operating Characteristic Curve (ROC AUC)
            st.write('Area Under ROC Curve : ' , roc_auc_score(y_test_binarize, y_pred_binarize, average='weighted', multi_class='ovo').round(2))

            #Confusion Matrix
            st.write('Confusion Matrix :')
            cfm = confusion_matrix(y_test, y_pred, normalize='true')
            df_cfm = pd.DataFrame(cfm, index = ['Abandoned', 'Cleared', 'Reexport', 'Seized'], columns = ['Abandoned', 'Cleared', 'Reexport', 'Seized'])
            fig, ax = plt.subplots(figsize=(10,6))
            plt.rcParams.update({'font.size': 12})
            ax = sns.heatmap(df_cfm, annot=True, cmap=plt.cm.Blues)
            ax.set(xlabel="Predicted Label", ylabel="True Label")
            st.pyplot(fig)
        
        def globalFeatureImportances(feature_importances, attributes, linear_model=True):
            importances = pd.DataFrame(data={
                                        'Attribute': attributes,
                                        'Importance': feature_importances})
            importances = importances.sort_values(by='Importance', ascending=False)
            num_importances = importances[importances['Attribute'].isin(columns_to_scale)]
            cat_importances = importances[importances['Attribute'].isin(cat_one_hot_attribs)]
            txt_importances = importances[importances['Attribute'].isin(txt_vect_attribs)]
            if linear_model:
                top20 = importances.head(20)
                bot20 = importances.tail(20)
                importances_20 = top20.append(bot20, ignore_index=True)
                cat_top10 = cat_importances.head(10)
                cat_bot10 = cat_importances.tail(10)
                cat_importances_10 = cat_top10.append(cat_bot10, ignore_index=True)
                txt_top10 = txt_importances.head(10)
                txt_bot10 = txt_importances.tail(10)
                txt_importances_10 = txt_top10.append(txt_bot10, ignore_index=True)
                palette = 'bwr'
            else:
                importances_20 = importances[:20]
                cat_importances_10 = cat_importances[:10]
                txt_importances_10 = txt_importances[:10]
                palette = 'Blues_r'

            fig, ax = plt.subplots(figsize=(16,12))
            plt.rcParams.update({'font.size': 12})
            ax = sns.barplot(y='Attribute', x='Importance', data=importances_20, palette=palette)
            st.pyplot(fig)
            
            return num_importances, cat_importances_10, txt_importances_10

        def globalFeatureImportancesByType(filtered_importances, linear_model=True):

            if linear_model:
                palette = 'bwr'
            else:
                palette = 'Blues_r'

            fig, ax = plt.subplots()
            plt.rcParams.update({'font.size': 10})
            ax = sns.barplot(y='Attribute', x='Importance', data=filtered_importances, palette=palette)
            ax.set(ylabel='')
            st.pyplot(fig)

        def limeModelExplanation(model, X_train_subset, X_validation_or_holdout, y_validation_or_holdout, y_pred, attributes):

            target_dict = {0: 'Abandoned', 1: 'Cleared', 2: 'Reexport', 3: 'Siezed'}
            y_validation_or_holdout_labeled = [target_dict[num] for num in y_validation_or_holdout]
            y_pred_labeled = [target_dict[num] for num in y_pred]

            false_preds = np.argwhere((y_pred != y_validation_or_holdout)).flatten()
            idx  = random.choice(false_preds)

            explainer = lime_tabular.LimeTabularExplainer(X_train_subset, mode="classification",
                                                        class_names=['Abandoned', 'Cleared', 'Reexport', 'Seized'],
                                                        feature_names=attributes)
            st.write("Prediction : ", y_pred_labeled[idx])
            st.write("Actual :     ", y_validation_or_holdout_labeled[idx])

            explanation = explainer.explain_instance(X_validation_or_holdout[idx], model.predict_proba, top_labels=4, 
                                                    labels=['Abandoned', 'Cleared', 'Reexport', 'Seized'])

            exp_html = explanation.as_html()
            components.html(exp_html, height=1200)

        #Running full modeling pipeline except for which model
        X_train_sm, X_test, y_train_sm, y_test, lemis_holdout_x, lemis_holdout_y, \
            columns_to_scale, cat_one_hot_attribs, txt_vect_attribs, \
                dummy_clf, y_pred_dummy, y_test_binarize, y_pred_dummy_binarize = predictionPipeline()
        #Concatening features into one attributes list
        attributes = columns_to_scale + cat_one_hot_attribs + txt_vect_attribs

        st.sidebar.subheader("Percentage of Training Data to Train Model(s) On")
        train_subset = st.sidebar.slider("Percentage of training data to train model on", .1, 100.0, step=.1, value=.1, key="train_subset")
        
        count_training_data = int(X_train_sm.shape[0]*(train_subset/100))
        idx = np.random.choice(np.arange(X_train_sm.shape[0]), count_training_data, replace=False)
        X_train_subset = X_train_sm[idx]
        y_train_subset = y_train_sm[idx]
        
        st.sidebar.subheader("Choose Dimensionality Reduction Technique")
        dim_reduction = st.sidebar.radio("Model", ('None', 'TruncatedSVD'), index=0, key="dim_red")
        if dim_reduction != 'None':
            n_components = st.sidebar.slider("Number of components to keep", 1, 5, step=1, value=2, key='n_components')

        if dim_reduction == "TruncatedSVD":
            tsvd = TruncatedSVD(random_state=RANDOM_SEED, n_components=n_components)
            X_train_subset = tsvd.fit_transform(X_train_subset)
            X_test = tsvd.transform(X_test)
            lemis_holdout_x = tsvd.transform(lemis_holdout_x)
        else:
            pass

        st.sidebar.subheader("Choose Classifier")
        classifier = st.sidebar.selectbox("Model", ("Logistic Regression", "SGDClassifier", "Decision Tree", "Random Forest", "XGBoost"))

        if classifier == "Logistic Regression":
            st.sidebar.subheader("Model Hyperparameters")
            solver = st.sidebar.radio("Algorithm to use in the optimization problem", ('sag', 'saga'), key="solver")
            C = st.sidebar.number_input("Inverse of regularization strength; smaller values specify stronger regularization", 0.01, 100.0, value=1.0, key="C")
            max_iter = st.sidebar.slider("Maxiumum number of interations", 1000, 10000, step=100, value=1000, key="max_iter")
            st.sidebar.subheader("Error Analysis")
            learning_curves_view = st.sidebar.checkbox("Learning Curves", False, key="learn_curves")
            if dim_reduction == 'None':
                featimp_text = "Feature Importances"
                featimp_view = st.sidebar.checkbox(featimp_text, False, key="fimp")
            else:
                pass  
            st.sidebar.subheader("Additional Options")
            validation_show = st.sidebar.checkbox("Show results on Validation Data", False, key="validation")

            if st.sidebar.button("Run Model", key='run'):
                col1, col2 = st.beta_columns(2)
                
                with col1:
                    st.subheader("Logistic Regression Classifier Results")
                    lr_clf = LogisticRegression(random_state=RANDOM_SEED, multi_class='auto', solver=solver, C=C, max_iter=max_iter)

                    lr_clf = lr_clf.fit(X_train_subset, y_train_subset)
                    y_pred = lr_clf.predict(X_test)
                    y_pred_holdout = lr_clf.predict(lemis_holdout_x)

                    labels = [0,1,2,3]
                    y_test_binarize = label_binarize(y_test, classes=labels)
                    y_pred_binarize = label_binarize(y_pred, classes=labels)
                    lemis_holdout_y_binarize = label_binarize(lemis_holdout_y, classes=labels)
                    y_pred_holdout_binarize = label_binarize(y_pred_holdout, classes=labels)

                    #Evaluate model
                    st.write("Results on Holdout Set:")
                    modelEvaluationMetrics(lr_clf, lemis_holdout_y, lemis_holdout_y_binarize, y_pred_holdout, y_pred_holdout_binarize, lemis_holdout_x)
                    if validation_show:
                        st.write("Results on Validation Set:")
                        modelEvaluationMetrics(lr_clf, y_test, y_test_binarize, y_pred, y_pred_binarize, X_test)

                with col2:
                    st.subheader("Dummy Classifier Results")
                    
                    #Evaluate model
                    st.write("Results with Most Frequent Strategy:")
                    modelEvaluationMetrics(dummy_clf, y_test, y_test_binarize, y_pred_dummy, y_pred_dummy_binarize, X_test)

                if learning_curves_view:
                    plotLearningCurves(lr_clf, X_train_subset, y_train_subset)

                if featimp_view:
                    st.subheader("Global Feature Importances:")
                    feature_importances = lr_clf.coef_[0]
                    num_importances, cat_importances, txt_importances = globalFeatureImportances(feature_importances, attributes)

                    col1, col2, col3 = st.beta_columns(3)
                    with col1:
                        globalFeatureImportancesByType(num_importances)
                    with col2:
                        globalFeatureImportancesByType(cat_importances)
                    with col3:
                        globalFeatureImportancesByType(txt_importances)

                    st.subheader("Local Feature Importances for Random Wrong Prediction:")
                    limeModelExplanation(lr_clf, X_train_subset, lemis_holdout_x, lemis_holdout_y, y_pred_holdout, attributes)

        if classifier == 'SGDClassifier':
            st.sidebar.subheader("Model Hyperparameters")
            loss = st.sidebar.radio("The loss function to be used", ('log', 'modified_huber'), key="loss")
            l1_ratio = st.sidebar.number_input("Ratio of L1 (value of 1)/L2 (value of 0) Regularization", 0.0, 1.0, step=0.1, value=.15, key="L1_L2")
            alpha = st.sidebar.select_slider("Constant that multiplies the regularization term", options=list(10.0**-np.arange(1,7)), key="alpha")
            max_iter = st.sidebar.number_input("The maximum number of passes over the training data (aka epochs)", 1000, 5000, step=100, value=1000, key="max_iter")
            early_stopping = st.sidebar.radio("Whether to use early stopping to terminate training when validation score is not improving", ('True', 'False'), key="estop")
            if early_stopping == 'True':
                early_stopping = True
            else:
                early_stopping = False
            st.sidebar.subheader("Error Analysis")
            learning_curves_view = st.sidebar.checkbox("Learning Curves", False, key="learn_curves")
            if dim_reduction == 'None':
                featimp_text = "Feature Importances"
                featimp_view = st.sidebar.checkbox(featimp_text, False, key="fimp")
            else:
                pass  
            st.sidebar.subheader("Additional Options")
            validation_show = st.sidebar.checkbox("Show results on Validation Data", False, key="validation")

            if st.sidebar.button("Run Model", key='run'):
                col1, col2 = st.beta_columns(2)

                with col1:
                    st.subheader("Stochastic Gradient Descent Classifier Results")
                    sgd_clf = SGDClassifier(random_state=RANDOM_SEED, penalty='elasticnet', learning_rate='adaptive', loss=loss,
                                            shuffle=True, l1_ratio=l1_ratio, alpha=alpha, max_iter=max_iter, eta0=1, early_stopping=early_stopping)

                    sgd_clf = sgd_clf.fit(X_train_subset, y_train_subset)
                    y_pred = sgd_clf.predict(X_test)
                    y_pred_holdout = sgd_clf.predict(lemis_holdout_x)

                    labels = [0,1,2,3]
                    y_test_binarize = label_binarize(y_test, classes=labels)
                    y_pred_binarize = label_binarize(y_pred, classes=labels)
                    lemis_holdout_y_binarize = label_binarize(lemis_holdout_y, classes=labels)
                    y_pred_holdout_binarize = label_binarize(y_pred_holdout, classes=labels)

                    #Evaluate model
                    st.write("Results on Holdout Set:")
                    modelEvaluationMetrics(sgd_clf, lemis_holdout_y, lemis_holdout_y_binarize, y_pred_holdout, y_pred_holdout_binarize, lemis_holdout_x)
                    if validation_show:
                        st.write("Results on Validation Set:")
                        modelEvaluationMetrics(sgd_clf, y_test, y_test_binarize, y_pred, y_pred_binarize, X_test)

                with col2:
                    st.subheader("Dummy Classifier Results")
                    
                    #Evaluate model
                    st.write("Results with Most Frequent Strategy:")
                    modelEvaluationMetrics(dummy_clf, y_test, y_test_binarize, y_pred_dummy, y_pred_dummy_binarize, X_test)

                if learning_curves_view:
                    plotLearningCurves(sgd_clf, X_train_subset, y_train_subset)

                if featimp_view:
                    st.subheader("Global Feature Importances:")
                    feature_importances = sgd_clf.coef_[0]
                    num_importances, cat_importances, txt_importances = globalFeatureImportances(feature_importances, attributes)

                    col1, col2, col3 = st.beta_columns(3)
                    with col1:
                        globalFeatureImportancesByType(num_importances)
                    with col2:
                        globalFeatureImportancesByType(cat_importances)
                    with col3:
                        globalFeatureImportancesByType(txt_importances)

                    st.subheader("Local Feature Importances for Random Wrong Prediction:")
                    limeModelExplanation(sgd_clf, X_train_subset, lemis_holdout_x, lemis_holdout_y, y_pred_holdout, attributes)

        if classifier == 'Decision Tree':
            st.sidebar.subheader("Model Hyperparameters")
            criterion = st.sidebar.radio("The function to measure the quality of a split", ('gini', 'entropy'), index=1, key="criterion")
            max_depth = st.sidebar.number_input("The maximum depth of the tree", 1, 50, step=1, value=20, key="max_depth")
            max_features = st.sidebar.radio("The number of features to consider when looking for the best split", ('sqrt', 'log2'), index=0, key="max_feat")
            st.sidebar.subheader("Error Analysis")
            learning_curves_view = st.sidebar.checkbox("Learning Curves", False, key="learn_curves")
            if dim_reduction == 'None':
                featimp_text = "Feature Importances"
                featimp_view = st.sidebar.checkbox(featimp_text, False, key="fimp")
            else:
                pass  
            st.sidebar.subheader("Additional Options")
            validation_show = st.sidebar.checkbox("Show results on Validation Data", False, key="validation")

            if st.sidebar.button("Run Model", key='run'):
                col1, col2 = st.beta_columns(2)

                with col1:
                    st.subheader("Decision Tree Classifier Results")
                    dt_clf = DecisionTreeClassifier(random_state=RANDOM_SEED, criterion=criterion, max_depth=max_depth, max_features=max_features)

                    dt_clf = dt_clf.fit(X_train_subset, y_train_subset)
                    y_pred = dt_clf.predict(X_test)
                    y_pred_holdout = dt_clf.predict(lemis_holdout_x)

                    labels = [0,1,2,3]
                    y_test_binarize = label_binarize(y_test, classes=labels)
                    y_pred_binarize = label_binarize(y_pred, classes=labels)
                    lemis_holdout_y_binarize = label_binarize(lemis_holdout_y, classes=labels)
                    y_pred_holdout_binarize = label_binarize(y_pred_holdout, classes=labels)

                    #Evaluate model
                    st.write("Results on Holdout Set:")
                    modelEvaluationMetrics(dt_clf, lemis_holdout_y, lemis_holdout_y_binarize, y_pred_holdout, y_pred_holdout_binarize, lemis_holdout_x)
                    if validation_show:
                        st.write("Results on Validation Set:")
                        modelEvaluationMetrics(dt_clf, y_test, y_test_binarize, y_pred, y_pred_binarize, X_test)

                with col2:
                    st.subheader("Dummy Classifier Results")
                    
                    #Evaluate model
                    st.write("Results with Most Frequent Strategy:")
                    modelEvaluationMetrics(dummy_clf, y_test, y_test_binarize, y_pred_dummy, y_pred_dummy_binarize, X_test)

                if learning_curves_view:
                    plotLearningCurves(dt_clf, X_train_subset, y_train_subset)

                if featimp_view:
                    st.subheader("Global Feature Importances:")
                    feature_importances = dt_clf.feature_importances_
                    num_importances, cat_importances, txt_importances = globalFeatureImportances(feature_importances, attributes, False)

                    col1, col2, col3 = st.beta_columns(3)
                    with col1:
                        globalFeatureImportancesByType(num_importances, False)
                    with col2:
                        globalFeatureImportancesByType(cat_importances, False)
                    with col3:
                        globalFeatureImportancesByType(txt_importances, False)

                    st.subheader("Local Feature Importances for Random Wrong Prediction:")
                    limeModelExplanation(dt_clf, X_train_subset, lemis_holdout_x, lemis_holdout_y, y_pred_holdout, attributes) 

        if classifier == 'Random Forest':
            st.sidebar.subheader("Model Hyperparameters")
            n_estimators = st.sidebar.number_input("The number of trees in the forest", 100, 5000, step=10, key='n_estimators')
            max_depth = st.sidebar.number_input("The maximum depth of the tree", 1, 100, step=1, key='max_depth')
            max_features = st.sidebar.selectbox("The number of features to consider when looking for best split", ('sqrt', 'log2'), index=0, key='max_feats')
            bootstrap = st.sidebar.radio("Bootstrap samples when building trees", ('True', 'False'), key='bootstrap')
            st.sidebar.subheader("Error Analysis")
            learning_curves_view = st.sidebar.checkbox("Learning Curves", False, key="learn_curves")
            if dim_reduction == 'None':
                featimp_text = "Feature Importances"
                featimp_view = st.sidebar.checkbox(featimp_text, False, key="fimp")
            else:
                pass  
            st.sidebar.subheader("Additional Options")
            validation_show = st.sidebar.checkbox("Show results on Validation Data", False, key="validation")

            if st.sidebar.button("Run Model", key='run'):
                col1, col2 = st.beta_columns(2)
                            
                with col1:
                    st.subheader("Random Forest Classifier Results")
                    rf_clf = RandomForestClassifier(random_state=RANDOM_SEED, max_depth=max_depth, max_features=max_features, bootstrap=bootstrap)

                    rf_clf = rf_clf.fit(X_train_subset, y_train_subset)
                    y_pred = rf_clf.predict(X_test)
                    y_pred_holdout = rf_clf.predict(lemis_holdout_x)

                    labels = [0,1,2,3]
                    y_test_binarize = label_binarize(y_test, classes=labels)
                    y_pred_binarize = label_binarize(y_pred, classes=labels)
                    lemis_holdout_y_binarize = label_binarize(lemis_holdout_y, classes=labels)
                    y_pred_holdout_binarize = label_binarize(y_pred_holdout, classes=labels)

                    #Evaluate model
                    st.write("Results on Holdout Set:")
                    modelEvaluationMetrics(rf_clf, lemis_holdout_y, lemis_holdout_y_binarize, y_pred_holdout, y_pred_holdout_binarize, lemis_holdout_x)
                    if validation_show:
                        st.write("Results on Validation Set:")
                        modelEvaluationMetrics(rf_clf, y_test, y_test_binarize, y_pred, y_pred_binarize, X_test)

                with col2:
                    st.subheader("Dummy Classifier Results")
                    
                    #Evaluate model
                    st.write("Results with Most Frequent Strategy:")
                    modelEvaluationMetrics(dummy_clf, y_test, y_test_binarize, y_pred_dummy, y_pred_dummy_binarize, X_test)

                if learning_curves_view:
                    plotLearningCurves(rf_clf, X_train_subset, y_train_subset)

                if featimp_view:
                    st.subheader("Global Feature Importances:")
                    feature_importances = rf_clf.feature_importances_
                    num_importances, cat_importances, txt_importances = globalFeatureImportances(feature_importances, attributes, False)

                    col1, col2, col3 = st.beta_columns(3)
                    with col1:
                        globalFeatureImportancesByType(num_importances, False)
                    with col2:
                        globalFeatureImportancesByType(cat_importances, False)
                    with col3:
                        globalFeatureImportancesByType(txt_importances, False)

                    st.subheader("Local Feature Importances for Random Wrong Prediction:")
                    limeModelExplanation(rf_clf, X_train_subset, lemis_holdout_x, lemis_holdout_y, y_pred_holdout, attributes) 

        if classifier == 'XGBoost':
            st.sidebar.subheader("Model Hyperparameters")
            eta = st.sidebar.number_input("Step size shrinkage used in update to prevents overfitting (learning rate)", 0.00, 1.00, step=0.05, value=0.3, key="eta")
            subsample = st.sidebar.number_input("Subsample ratio of the training instances", 0.0, 1.0, step=0.1, value=0.5, key="subsamp")
            max_depth = st.sidebar.number_input("The maximum depth of the tree", 1, 100, step=1, key='max_depth')
            reg_lambda = st.sidebar.number_input("L2 regularization term on weights", 0.0, 1.0, step=0.1, value=1.0, key='lambda')
            st.sidebar.subheader("Error Analysis")
            learning_curves_view = st.sidebar.checkbox("Learning Curves", False, key="learn_curves")
            if dim_reduction == 'None':
                featimp_text = "Feature Importances"
                featimp_view = st.sidebar.checkbox(featimp_text, False, key="fimp")
            else:
                pass            
            st.sidebar.subheader("Additional Options")
            validation_show = st.sidebar.checkbox("Show results on Validation Data", False, key="validation")

            if st.sidebar.button("Run Model", key='run'):
                col1, col2 = st.beta_columns(2)

                with col1:
                    st.subheader("XGBoost Classifier Results")
                    xgb_clf = xgb.XGBClassifier(random_state=RANDOM_SEED, objective='multi:softprob', use_label_encoder=False, 
                                                eta=eta, subsample=subsample, max_depth=max_depth, reg_lambda=reg_lambda)

                    xgb_clf = xgb_clf.fit(X_train_subset, y_train_subset)
                    y_pred = xgb_clf.predict(X_test)
                    y_pred_holdout = xgb_clf.predict(lemis_holdout_x)

                    labels = [0,1,2,3]
                    y_test_binarize = label_binarize(y_test, classes=labels)
                    y_pred_binarize = label_binarize(y_pred, classes=labels)
                    lemis_holdout_y_binarize = label_binarize(lemis_holdout_y, classes=labels)
                    y_pred_holdout_binarize = label_binarize(y_pred_holdout, classes=labels)

                    #Evaluate model
                    st.write("Results on Holdout Set:")
                    modelEvaluationMetrics(xgb_clf, lemis_holdout_y, lemis_holdout_y_binarize, y_pred_holdout, y_pred_holdout_binarize, lemis_holdout_x)
                    if validation_show:
                        st.write("Results on Validation Set:")
                        modelEvaluationMetrics(xgb_clf, y_test, y_test_binarize, y_pred, y_pred_binarize, X_test)

                with col2:
                    st.subheader("Dummy Classifier Results")
                    
                    #Evaluate model
                    st.write("Results with Most Frequent Strategy:")
                    modelEvaluationMetrics(dummy_clf, y_test, y_test_binarize, y_pred_dummy, y_pred_dummy_binarize, X_test)

                if learning_curves_view:
                    plotLearningCurves(xgb_clf, X_train_subset, y_train_subset)

                if featimp_view:
                    st.subheader("Global Feature Importances:")
                    feature_importances = xgb_clf.feature_importances_
                    num_importances, cat_importances, txt_importances = globalFeatureImportances(feature_importances, attributes, False)

                    col1, col2, col3 = st.beta_columns(3)
                    with col1:
                        globalFeatureImportancesByType(num_importances, False)
                    with col2:
                        globalFeatureImportancesByType(cat_importances, False)
                    with col3:
                        globalFeatureImportancesByType(txt_importances, False)

                    st.subheader("Local Feature Importances for Random Wrong Prediction:")
                    limeModelExplanation(xgb_clf, X_train_subset, lemis_holdout_x, lemis_holdout_y, y_pred_holdout, attributes)

        st.markdown("This [data set](https://data.nal.usda.gov/dataset/data-united-states-wildlife-and-wildlife-product-imports-2000%E2%80%932014) \
                    includes data on 15 years of the importation of wildlife and their derived products into the United States (2000–2014), \
                    originally collected by the United States Fish and Wildlife Service. "
                    "The data used in this application is only about 1% (total, before filtering to some percentage of training data) of all data available \
                    due to Github and Streamlit size limitations.")

    elif option == "Cluster Shipments from Panjiva":
        #Top title on the sidebar
        st.sidebar.title("Clustering Shipments from Panjiva")

        nlp = en_core_web_sm.load()
        stop_words = spacy.lang.en.stop_words.STOP_WORDS
        punctuations = string.punctuation
        parser = English()

        @st.cache(persist=True, suppress_st_warning=True, allow_output_mutation=True)
        def featureCleaning(panjiva):
            """
            Input:
                panjiva = panjiva dataframe
            Output:
                panjiva_engineered = panjiva dataframe with features cleaned/engineered and appropriate columns
                goods_shipped = vector of goods shipped
            """

            #Create a feature for the perc of columns missing
            panjiva['miss_perc'] = panjiva.isnull().sum(axis=1).tolist()
            panjiva['miss_perc'] = panjiva.miss_perc/(len(panjiva.columns)-1)
            panjiva['goods shipped'] = panjiva['goods shipped'].apply(lambda x: '_'+str(x))
            panjiva['goods shipped'] = panjiva['goods shipped'].dropna()

            panjiva['lading_cat'] = panjiva['port of lading'].astype('category').cat.codes
            panjiva['unlading_cat'] = panjiva['port of unlading'].astype('category').cat.codes
            panjiva['consignee_cat'] = panjiva['consignee'].astype('category').cat.codes
            panjiva['transport_cat'] = panjiva['transport method'].astype('category').cat.codes
            panjiva['container_cat'] = panjiva['is containerized'].astype('category').cat.codes
            panjiva['hscode_cat'] = panjiva['hscode'].astype('category').cat.codes
            panjiva_engineered = panjiva[['volume (teu)', 'weight (kg)','miss_perc', 
                                            'value (usd)', 'hscode_cat', 'lading_cat', 
                                            'unlading_cat', 'consignee_cat','container_cat']]
            
            std_scaler = StandardScaler()
            panjiva_engineered[['volume (teu)', 'weight (kg)', 'value (usd)']] = std_scaler.fit_transform(panjiva_engineered[['volume (teu)', 'weight (kg)', 'value (usd)']])
            panjiva_engineered = panjiva_engineered.fillna(-2)

            return panjiva_engineered, panjiva['goods shipped']
        
        @st.cache(persist=True, suppress_st_warning=True, allow_output_mutation=True)
        def clusteringPipeline():
            """
            Input:
                None
            Output:
                panjiva_engineered = panjiva dataframe with features cleaned/engineered and appropriate columns
                goods_shipped = vector of goods shipped
            """

            #Load in local file and conduct feature engineering
            panjiva = loadData('Panjiva')
            panjiva_engineered, goods_shipped = featureCleaning(panjiva)

            return panjiva_engineered, goods_shipped

        def spacy_tokenizer(sent):

            toks = parser(sent)
            toks = [ word for word in toks if word not in stop_words ]
            toks = [ word for word in toks if word.text.isdigit() == False]
            toks = [ word for word in toks if len(word.text) > 1]
            toks = [word.text for word in toks]
            toks = [word.replace('\n', '').replace('\r', '').replace(' ','') for word in toks]
            toks = [word for word in toks if any(map(str.isdigit, word)) == False]

            return toks

        def splomchart(panjiva_engineered_subset, hue):
            
            num_cols = ['volume (teu)', 'weight (kg)', 'value (usd)', 'miss_perc', hue]
            fig = sns.pairplot(panjiva_engineered_subset[num_cols], hue=hue, diag_kind="kde", corner=True)
            st.pyplot(fig)

        def elbowPlot(model_name, goods_shipped_vect, elbow_clusters):
            if model_name == 'KPrototypes':

                K = range(1,elbow_clusters)
                cost = []
                for k in K:
                    model = KPrototypes(random_state=RANDOM_SEED, n_clusters=k, init='Huang')
                    model.fit_predict(goods_shipped_vect, categorical = [4,5,6,7,8])
                    cost.append(model.cost_)

                fig, ax = plt.subplots(figsize=(24,10))
                ax = sns.lineplot(x=cost, y=K)
                ax.set(xlabel='Values of K', ylabel='Cost', title='The Elbow Method using Cost')
                st.pyplot(fig)

            elif model_name == 'KMeans':
                K = range(1,elbow_clusters)
                distortions = []
                inertias = []
                X = goods_shipped_vect
                for k in K:
                    model = KMeans(random_state=RANDOM_SEED, n_clusters=k)
                    model.fit_predict(X)
                    #distortions.append(sum(np.min(cdist(X, model.cluster_centers_, 'euclidean'), axis=1)) / X.shape[0])

                    inertias.append(model.inertia_)

                fig, ax = plt.subplots(figsize=(12,10))
                ax = sns.lineplot(x=inertias, y=K)
                ax.set(xlabel='Values of K', ylabel='Inertia', title='The Elbow Method using Inertia')
                st.pyplot(fig)

        #helper function
        def black_color_func(word, font_size, position,orientation,random_state=None, **kwargs):
            return("hsl(0,100%, 1%)")

        def plotWordcloud(panjiva_engineered_subset, cluster_col, cluster, tokenized_goods_col):
            
            df = panjiva_engineered_subset[panjiva_engineered_subset[cluster_col] == cluster]

            #could loop through this, but not really needed
            w = WordCloud(width=800,height=600,mode='RGBA',background_color='white',max_words=50
                            ).generate(' '.join(df[tokenized_goods_col].sum()))

            # set the word color to black
            w.recolor(color_func=black_color_func)
            # set the figsize
            fig, ax = plt.subplots(figsize=(12,10))
            # plot the wordcloud
            ax.imshow(w, interpolation="bilinear")
            # remove plot axes
            ax.axis("off")
            st.pyplot(fig)

        panjiva_engineered, goods_shipped = clusteringPipeline()

        st.sidebar.subheader("Percentage of Training Data to Train Model(s) On")
        train_subset = st.sidebar.slider("Percentage of training data to train model on", .1, 100.0, step=.1, value=.1, key="train_subset")
        
        panjiva_engineered_subset = panjiva_engineered.sample(random_state=RANDOM_SEED, frac=train_subset/100)
        goods_shipped_subset = goods_shipped[panjiva_engineered_subset.index.tolist()]
        
        st.sidebar.subheader("Choose Clustering Technique")
        clustering = st.sidebar.selectbox("Model", ("KMeans", "KPrototypes"))

        if clustering == "KMeans":
            st.sidebar.write("This clustering is done only on the goods shipped column")
            st.sidebar.subheader("Choose Text Vectorizer Method")
            vectorizer = st.sidebar.multiselect("Text Vectorizer", ["CountVectorizer", "TfidfVectorizer"], default=["CountVectorizer"])
            st.sidebar.subheader("Model Hyperparameters")
            n_clusters = st.sidebar.number_input("The number of clusters to form as well as the number of centroids to generate", 2, 20, step=1, value=8, key="clusters")
            init = st.sidebar.radio("Method for initialization", ('k-means++', 'random'), index=0, key="init")
            st.sidebar.subheader("Error Analysis")
            elbow_plot_view = st.sidebar.checkbox("Elbow Plot", False, key="elbow_plot")
            if elbow_plot_view:
                elbow_clusters = st.sidebar.slider("The number of clusters to try out", 1,100, step=1, key="elbow_clusters")
            st.sidebar.subheader("Additional Options")
            wordcloud_option = st.sidebar.checkbox("WordCloud for a Cluster", False, key="wordcloud")

            cluster = random.randint(0, n_clusters)
            cluster = cluster-1
            if st.sidebar.button("Run Model", key='run'):
                st.subheader("KMeans Clustering Results")
                kmean_clust = KMeans(n_clusters=n_clusters, init=init)
                
                col1, col2 = st.beta_columns(2)
                
                if "CountVectorizer" in vectorizer:
                    with col1:
                        st.write("Results with CountVectorizer:")
                        count_vectorizer = CountVectorizer(ngram_range=(1,2), tokenizer=spacy_tokenizer)
                        goods_cv = count_vectorizer.fit_transform(goods_shipped_subset)
                        goods_count_nn = NearestNeighbors(metric='cosine', algorithm='brute')
                        goods_count_nn.fit(goods_cv)
                        
                        pred = kmean_clust.fit_predict(goods_cv)
                        panjiva_engineered_subset['cluster_countvect'] = pred

                        splomchart(panjiva_engineered_subset, "cluster_countvect")

                        if elbow_plot_view:
                            st.subheader("Elbow Plot:")
                            elbowPlot('KMeans', goods_cv, elbow_clusters)
                
                        if wordcloud_option:
                            st.subheader(f"WordCloud for Random Cluster {cluster}:")
                            #Tokenize the strings
                            panjiva_engineered_subset['count_tokenized_goods'] = goods_shipped_subset.apply(lambda x :spacy_tokenizer(x))
                            
                            plotWordcloud(panjiva_engineered_subset, "cluster_countvect", cluster, "count_tokenized_goods")

                if "TfidfVectorizer" in vectorizer:
                    with col2:
                        st.write("Results with TfidfVectorizer:")
                        tfidf_vectorizer = TfidfVectorizer(input='content', tokenizer=spacy_tokenizer)
                        goods_tfidf = tfidf_vectorizer.fit_transform(goods_shipped_subset)
                        goods_tfidf_nn = NearestNeighbors(metric='cosine', algorithm='brute')
                        goods_tfidf_nn.fit(goods_tfidf)
                        
                        pred = kmean_clust.fit_predict(goods_tfidf)
                        panjiva_engineered_subset["cluster_tfidfvect"] = pred

                        splomchart(panjiva_engineered_subset, "cluster_tfidfvect")

                        if elbow_plot_view:
                            st.subheader("Elbow Plot:")
                            elbowPlot('KMeans', goods_tfidf, elbow_clusters)

                        if wordcloud_option:
                            st.subheader(f"WordCloud for Random Cluster {cluster}:")
                            #Tokenize the strings
                            panjiva_engineered_subset['tfidf_tokenized_goods'] = goods_shipped_subset.apply(lambda x :spacy_tokenizer(x))

                            plotWordcloud(panjiva_engineered_subset, "cluster_tfidfvect", cluster, "tfidf_tokenized_goods")


        if clustering == "KPrototypes":
            st.sidebar.write("This clustering is done on all columns with mixed data types")
            st.sidebar.subheader("Model Hyperparameters")
            n_clusters = st.sidebar.number_input("The number of clusters to form as well as the number of centroids to generate", 2, 20, step=1, value=8, key="clusters")
            st.sidebar.subheader("Error Analysis")
            elbow_plot_view = st.sidebar.checkbox("Elbow Plot", False, key="elbow_plot")
            if elbow_plot_view:
                elbow_clusters = st.sidebar.slider("The number of clusters to try out", 1,100, step=1, key="elbow_clusters")

            if st.sidebar.button("Run Model", key='run'):
                st.subheader("KPrototypes Clustering Results")
                kprot_clust = KPrototypes(n_clusters=n_clusters, init='Huang')
                pred = kprot_clust.fit_predict(panjiva_engineered_subset, categorical = [4,5,6,7,8])

                panjiva_engineered_subset['cluster'] = pred
                
                splomchart(panjiva_engineered_subset, "cluster")
                
                if elbow_plot_view:
                    st.subheader("Elbow Plot:")
                    elbowPlot('KPrototypes', panjiva_engineered_subset, elbow_clusters)

        st.markdown("This data set was manually downloaded through a paid [Panjiva](https://panjiva.com/) account \
                    and it includes data for imported shipments (2007-2021) related to wildlife for HS codes 01, 02, 03, 04, and 05 as these represent animals & animal products. "
                    "The data used in this application is only about 1% (total, before filtering to some percentage of training data) of all data available \
                    due to Github and Streamlit size limitations.")

if __name__ == '__main__':
    main()
