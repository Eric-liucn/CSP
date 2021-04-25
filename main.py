'''
In this pipeline, it first carried out a relatively complete data cleaning, including processing missing values
、and outliers, removing useless columns, and scaling operations. Then, it uses two approaches to implement feature
selection, the first is to use the SelectKBest model provided by Scikit-learn, and the second is the RFE model,
which uses a random forest classifier as the estimator. The pipeline retains the features selected by the two results.
Finally, the pipeline uses PCA (Principal Component Analysis) to reduce the dimensionality of the data.
'''

import numpy as np
import pandas as pd
from scipy.stats.mstats import winsorize
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE, SelectKBest, mutual_info_classif
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class DataPipeline(object):

    def __init__(self, data_path: str) -> None:
        super().__init__()
        # read data from given data file path and use the column 0 as index
        self.selected_features = []
        self.df = pd.read_csv(data_path, index_col=0)
        # do data clean work
        self.data_clean()
        # Use SelectKBest to do feature selection
        self.feature_selection_selectkbest()
        # Use RFE to select features and keep that with the result of SelectKBest
        self.feature_selection_rfe()
        # use PCA to reduce dimensionality
        self.do_principal_component_analysis()

    def data_clean(self):
        # Drop useless columns "filename" and "length"
        # "filename" can not be used to analysis and "length" are same for all data entries
        self.df = self.df.drop(['filename', 'length'], axis=1)

        # Fill missing value with median
        # Adopted from: https://towardsdatascience.com/data-cleaning-in-python-the-ultimate-guide-2020-c63b88bf0a0d
        for column in self.df.columns:
            missing = self.df[column].isnull()
            num_missing = np.sum(missing)
            if num_missing > 0:
                self.df[column].fillna(self.df[column].median())
                print("Fill missing value of column {} with the median of column {}".format(column, column))
            else:
                print("No missing value in column {}".format(column))

        # winsorize remove outliers, make the data smoother
        # Adopted from: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.mstats.winsorize.html
        for column in self.df.columns:
            self.df[column] = winsorize(self.df[column], limits=[0.05, 0.05])

        # Convert String type data to Integer type for further analysis
        mapping_dict = {
            'blues': 1,
            'classical': 2,
            'country': 3,
            'disco': 4,
            'hiphop': 5,
            'jazz': 6,
            'metal': 7,
            'pop': 8,
            'reggae': 9,
            'rock': 10
        }
        self.df['label'] = self.df['label'].map(mapping_dict)

        # Feature Scaling by using Standard Scalar
        # adopted from: https://stackoverflow.com/questions/35723472/how-to-use-sklearn-fit-transform-with-pandas-and-return-dataframe-instead-of-num
        self.df = pd.DataFrame(StandardScaler().fit_transform(self.df), index=self.df.index, columns=self.df.columns)

    def feature_selection_selectkbest(self):
        # Use SelectKBest to choose feature
        y = self.df['label']
        x = self.df.drop("label", axis=1)
        # Consider that the "label" were string data, so try to use classifier
        selector = SelectKBest(score_func=mutual_info_classif, k=11)
        x_new = selector.fit_transform(x, y.astype("int"))
        print('+-------------------------------------+')
        print('+          feature selection          +')
        print('+-------------------------------------+')
        for index in selector.get_support(True):
            print("selected feature by SelectKBest: {}".format(self.df.columns[index]))
            # keep the features selected by SelectKBest
            self.selected_features.append(self.df.columns[index])
        # save the feature selection result of KBest

    def feature_selection_rfe(self):
        y = self.df['label']
        x = self.df.drop("label", axis=1)
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=101)
        num_of_features_list = np.arange(5, 55)
        # use random forest classifier as the estimator of RFE
        rand_classifier = RandomForestClassifier()
        score_list = []
        highest_score = 0
        num_of_features = 0
        # Try different number of features to get the best result
        for i in range(len(num_of_features_list)):
            selector = RFE(rand_classifier, n_features_to_select=num_of_features_list[i], verbose=10)
            x_train_rfe = selector.fit_transform(x_train, y_train.astype("int"))
            x_test_rfe = selector.transform(x_test)
            rand_classifier.fit(x_train_rfe, y_train.astype("int"))
            score = rand_classifier.score(x_test_rfe, y_test.astype("int"))
            score_list.append(score)
            if score > highest_score:
                highest_score = score
                num_of_features = num_of_features_list[i]
        print("The highest score is {}, with features number of {}".format(highest_score, num_of_features))

        # use the best choice of number of features to fit RFE model
        selector = RFE(rand_classifier, n_features_to_select=num_of_features, verbose=10)
        selector.fit(x_train, y_train.astype("int"))
        for index in selector.get_support(True):
            name = self.df.columns[index]
            print("selected feature by RFE: {}".format(name))
            # keep both features selected by KBest & REF
            if name not in self.selected_features:
                self.selected_features.append(name)

    def do_principal_component_analysis(self):
        self.selected_features.append("label")
        df_selected_feature = self.df[self.selected_features]
        # adopted from: https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html
        pca = PCA(0.8)
        print(df_selected_feature.shape)
        x_pca = pca.fit_transform(df_selected_feature)
        print(x_pca.shape)


if __name__ == '__main__':
    pipeline = DataPipeline("data/music_audio_data.csv")
