import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.impute import KNNImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler

from project_library_layer.initializer.initializer import Initializer
from exception_layer.generic_exception.generic_exception import GenericException as PreprocessorException
import sys
from imblearn.over_sampling import RandomOverSampler
from sklearn_pandas import CategoricalImputer
from imblearn.over_sampling import SMOTE
from plotly_dash.accuracy_graph.accuracy_graph import AccurayGraph


class Preprocessor:
    """
        This class shall  be used to clean and transform the data before training.

        Written By: iNeuron Intelligence
        Version: 1.0
        Revisions: None

        """

    def __init__(self, file_object, logger_object, project_id):
        try:
            self.file_object = file_object
            self.logger_object = logger_object
            self.project_id = project_id
            self.initializer = Initializer()
        except Exception as e:
            pre_processor_exception = PreprocessorException(
                "Failed during object instantiation in module [{0}] class [{1}] method [{2}]"
                    .format(self.__module__, Preprocessor.__name__,
                            self.__init__.__name__))
            raise Exception(pre_processor_exception.error_message_detail(str(e), sys)) from e

    def handle_imbalance_dataset_forest_cover(self, X, Y):
        try:
            sample = SMOTE()
            X, Y = sample.fit_resample(X, Y)

            return X, Y
        except Exception as e:
            pre_processor_exception = PreprocessorException(
                "Failed  in module [{0}] class [{1}] method [{2}]"
                    .format(self.__module__, Preprocessor.__name__,
                            self.handle_imbalance_dataset_forest_cover.__name__))
            raise Exception(pre_processor_exception.error_message_detail(str(e), sys)) from e

    def scale_data_forest_cover(self, data):
        try:
            self.logger_object.log("Started scaling data.")
            scalar = StandardScaler()
            num_data = data[
                ["elevation", "aspect", "slope", "horizontal_distance_to_hydrology", "Vertical_Distance_To_Hydrology",
                 "Horizontal_Distance_To_Roadways", "Horizontal_Distance_To_Fire_Points"]]
            cat_data = data.drop(
                ["elevation", "aspect", "slope", "horizontal_distance_to_hydrology", "Vertical_Distance_To_Hydrology",
                 "Horizontal_Distance_To_Roadways", "Horizontal_Distance_To_Fire_Points"], axis=1)
            scaled_data = scalar.fit_transform(num_data)
            num_data = pd.DataFrame(scaled_data, columns=num_data.columns, index=num_data.index)
            final_data = pd.concat([num_data, cat_data], axis=1)
            self.logger_object.log("Scaling completed")
            return final_data
        except Exception as e:
            pre_processor_exception = PreprocessorException(
                "Failed  in module [{0}] class [{1}] method [{2}]"
                    .format(self.__module__, Preprocessor.__name__,
                            self.scale_data_forest_cover.__name__))
            raise Exception(pre_processor_exception.error_message_detail(str(e), sys)) from e

    def encode_categorical_value_forest_cover(self, data):
        try:
            data["class"] = data["class"].map(
                {"Lodgepole_Pine": 0, "Spruce_Fir": 1, "Douglas_fir": 2, "Krummholz": 3, "Ponderosa_Pine": 4,
                 "Aspen": 5,
                 "Cottonwood_Willow": 6})
            return data
        except Exception as e:
            pre_processor_exception = PreprocessorException(
                "Failed during handling the imbalance in the dataset by oversampling. in module [{0}] class [{1}] method [{2}]"
                    .format(self.__module__, Preprocessor.__name__,
                            self.encode_categorical_value_forest_cover.__name__))
            raise Exception(pre_processor_exception.error_message_detail(str(e), sys)) from e

    def handle_imbalance_dataset(self, X, Y):
        """
        Method Name: handle_imbalance_dataset
        Description: This method handles the imbalance in the dataset by oversampling.
        Output: A Dataframe which is balanced now.
        On Failure: Raise Exception

        Written By: iNeuron Intelligence
        Version: 1.0
        Revisions: None
        """

        try:
            rd_sample = RandomOverSampler()
            x_sampled, y_sampled = rd_sample.fit_sample(X, Y)

            return x_sampled, y_sampled
        except Exception as e:
            pre_processor_exception = PreprocessorException(
                "Failed during handling the imbalance in the dataset by oversampling. in module [{0}] class [{1}] method [{2}]"
                    .format(self.__module__, Preprocessor.__name__,
                            self.handle_imbalance_dataset.__name__))
            raise Exception(pre_processor_exception.error_message_detail(str(e), sys)) from e

    def replace_invalid_values_with_null_fitbit(self, data):
        """

        :return:
        """
        try:
            for column in data.columns:
                count = data[column][data[column] == 'na'].count()
                if count != 0:
                    data[column] = data[column].replace('na', np.nan)
            return data
        except Exception as e:
            pre_processor_exception = PreprocessorException(
                "Failed during handling the imbalance in the dataset by oversampling. in module [{0}] class [{1}] method [{2}]"
                    .format(self.__module__, Preprocessor.__name__,
                            self.replace_invalid_values_with_null_fitbit.__name__))
            raise Exception(pre_processor_exception.error_message_detail(str(e), sys)) from e

    def remove_duplicate(self, data):
        """
                        Method Name: removeDuplicates
                        Description: This method removes the duplicates from the data

                        Written By: iNeuron Intelligence
                        Version: 1.0
                        Revisions: None

                                """
        try:
            self.logger_object.log("Started removing duplicate data")
            duplicate_count = data.duplicated().sum()
            if duplicate_count > 0:
                data.drop_duplicates(inplace=True)
            self.logger_object.log("Completed the process of removing duplicate data")
            return data
        except Exception as e:
            pre_processor_exception = PreprocessorException(
                "Failed in module [{0}] class [{1}] method [{2}]"
                    .format(self.__module__, Preprocessor.__name__,
                            self.remove_duplicate.__name__))
            raise Exception(pre_processor_exception.error_message_detail(str(e), sys)) from e

    def drop_unnecessary_columns(self, data, column_name_list):
        """
                        Method Name: is_null_present
                        Description: This method drops the unwanted columns as discussed in EDA section.

                        Written By: iNeuron Intelligence
                        Version: 1.0
                        Revisions: None

                                """
        try:
            self.logger_object.log("Entered the remove_columns method of the Preprocessor class")
            data = data.drop(column_name_list, axis=1)
            self.logger_object.log(
                'Column removal Successful.Exited the remove_columns method of the Preprocessor class')
            return data
        except Exception as e:
            pre_processor_exception = PreprocessorException(
                "Failed during droping  unwanted columns in module [{0}] class [{1}] method [{2}]"
                    .format(self.__module__, Preprocessor.__name__,
                            self.drop_unnecessary_columns.__name__))
            raise Exception(pre_processor_exception.error_message_detail(str(e), sys)) from e

    def remove_unwanted_spaces(self, data):
        """
                        Method Name: remove_unwanted_spaces
                        Description: This method removes the unwanted spaces from a pandas dataframe.
                        Output: A pandas DataFrame after removing the spaces.
                        On Failure: Raise Exception

                        Written By: iNeuron Intelligence
                        Version: 1.0
                        Revisions: None

                """

        try:
            self.logger_object.log('Entered the remove_unwanted_spaces method of the Preprocessor class')
            self.data = data
            self.df_without_spaces = self.data.apply(
                lambda x: x.str.strip() if x.dtype == "object" else x)  # drop the labels specified in the columns
            self.logger_object.log(
                'Unwanted spaces removal Successful.Exited the remove_unwanted_spaces method of the Preprocessor class')
            return self.df_without_spaces
        except Exception as e:
            pre_processor_exception = PreprocessorException(
                "Failed  in module [{0}] class [{1}] method [{2}]"
                    .format(self.__module__, Preprocessor.__name__,
                            self.remove_unwanted_spaces.__name__))
            raise Exception(pre_processor_exception.error_message_detail(str(e), sys)) from e

    def remove_columns(self, data, columns):
        """
                Method Name: remove_columns
                Description: This method removes the given columns from a pandas dataframe.
                Output: A pandas DataFrame after removing the specified columns.
                On Failure: Raise Exception

                Written By: iNeuron Intelligence
                Version: 1.0
                Revisions: None

        """

        try:
            self.logger_object.log('Entered the remove_columns method of the Preprocessor class')
            useful_data = data.drop(labels=columns, axis=1)  # drop the labels specified in the columns
            self.logger_object.log('Column removal Successful.Exited the remove_columns method of the Preprocessor '
                                   'class')
            return useful_data
        except Exception as e:
            pre_processor_exception = PreprocessorException(
                "Failed during column removal process in module [{0}] class [{1}] method [{2}]"
                    .format(self.__module__, Preprocessor.__name__,
                            self.remove_columns.__name__))
            raise Exception(pre_processor_exception.error_message_detail(str(e), sys)) from e

    def replace_invalid_values_with_null(self, data):

        """
                               Method Name: is_null_present
                               Description: This method replaces invalid values i.e. '?' with null, as discussed in EDA.

                               Written By: iNeuron Intelligence
                               Version: 1.0
                               Revisions: None

                                       """
        try:
            """
            data = data.replace("'?'", np.nan)
            data = data.apply(pd.to_numeric)
            """
            for column in data.columns:
                count = data[column][data[column] == "?"].count()
                if count != 0:
                    data[column] = data[column].replace("?", np.nan)
            return data

        except Exception as e:
            pre_processor_exception = PreprocessorException(
                "Failed during removing null string in module [{0}] class [{1}] method [{2}]"
                    .format(self.__module__, Preprocessor.__name__,
                            self.replace_invalid_values_with_null.__name__))
            raise Exception(pre_processor_exception.error_message_detail(str(e), sys)) from e

    def remove_null_string(self, data):
        """
        :param data: accpet data frame
        :return: replace NULL string to np.nan
        """
        try:
            data = data.replace("NULL", np.nan)
            data = data.apply(pd.to_numeric)
            return data
        except Exception as e:
            pre_processor_exception = PreprocessorException(
                "Failed during removing null string in module [{0}] class [{1}] method [{2}]"
                    .format(self.__module__, Preprocessor.__name__,
                            self.remove_null_string.__name__))
            raise Exception(pre_processor_exception.error_message_detail(str(e), sys)) from e

    def encode_categorical_values(self, data):
        """
        Method Name: encodeCategoricalValues
        Description: This method encodes all the categorical values in the training set.
        Output: A Dataframe which has all the categorical values encoded.
        On Failure: Raise Exception

        Written By: iNeuron Intelligence
        Version: 1.0
        Revisions: None
        """
        try:
            data['sex'] = data['sex'].map({'F': 0, 'M': 1})

            # except for 'Sex' column all the other columns with two categorical data have same value 'f' and 't'.
            # so instead of mapping indvidually, let's do a smarter work
            print(data.dtypes)
            for column in data.columns:
                if len(data[column].unique()) == 2:
                    data[column] = data[column].map({'f': 0, 't': 1})

            # this will map all the rest of the columns as we require. Now there are handful of column left with more than 2 categories.
            # we will use get_dummies with that.
            data = pd.get_dummies(data, columns=['referral_source'])

            encode = LabelEncoder().fit(data['Class'])

            data['Class'] = encode.transform(data['Class'])
            encoder_file_path = self.initializer.get_encoder_pickle_file_path(self.project_id)
            file_name = self.initializer.get_encoder_pickle_file_name()
            self.file_object.write_file_content(encoder_file_path, file_name, encode)
            return data
        except Exception as e:
            pre_processor_exception = PreprocessorException(
                "Failed during method encodes all the categorical values in the training set in module [{0}] class [{1}] method [{2}]"
                    .format(self.__module__, Preprocessor.__name__,
                            self.encode_categorical_values.__name__))
            raise Exception(pre_processor_exception.error_message_detail(str(e), sys)) from e

    def get_absent_column_name_mushroom(self, existing_column_names, new_column_names, data):
        try:
            for existing_col in existing_column_names:
                if existing_col not in new_column_names:
                    data[existing_col] = np.zeros(shape=(data.shape[0]))
            return data[existing_column_names]
        except Exception as e:
            pre_processor_exception = PreprocessorException(
                "Failed during remaining column generation in the training set in module [{0}] class [{1}] method [{2}]"
                    .format(self.__module__, Preprocessor.__name__,
                            self.get_absent_column_name_mushroom.__name__))
            raise Exception(pre_processor_exception.error_message_detail(str(e), sys)) from e

    def encode_categorical_values_prediction_mushroom(self, data):
        """
                                               Method Name: encodeCategoricalValuesPrediction
                                               Description: This method encodes all the categorical values in the prediction set.
                                               Output: A Dataframe which has all the categorical values encoded.
                                               On Failure: Raise Exception

                                               Written By: iNeuron Intelligence
                                               Version: 1.0
                                               Revisions: None
                            """
        try:
            for column in data.columns:
                data = pd.get_dummies(data, columns=[column], drop_first=True)
            response = self.file_object.read_file_content(
                self.initializer.get_encoded_column_name_file_path(self.project_id),
                self.initializer.get_encoded_column_file_name())
            if not response['status']:
                raise Exception("Column name not found while trying get it from cloud file.")

            column_names = response.get('file_content', None)
            if column_names is None:
                raise Exception("Column name not found while trying get it from cloud file")

            data = self.get_absent_column_name_mushroom(column_names, data.columns.values, data)
            return data
        except Exception as e:
            pre_processor_exception = PreprocessorException(
                "Failed during set in module [{0}] class [{1}] method [{2}]"
                    .format(self.__module__, Preprocessor.__name__,
                            self.encode_categorical_values_prediction_mushroom.__name__))
            raise Exception(pre_processor_exception.error_message_detail(str(e), sys)) from e

    def encode_categorical_columns_income_prediction(self, data):
        """
        Method Name: encode_categorical_columns
        Description: This method encodes the categorical values to numeric values.
        Output: only the columns with categorical values converted to numerical values
        On Failure: Raise Exception

        Written By: iNeuron Intelligence
        Version: 1.0
        Revisions: None
        """
        try:
            self.logger_object.log('Entered the encode_categorical_columns method of the Preprocessor class')
            self.cat_df = data.select_dtypes(include=['object']).copy()
            # Using the dummy encoding to encode the categorical columns to numericsl ones
            for col in self.cat_df.columns:
                self.cat_df = pd.get_dummies(self.cat_df, columns=[col], prefix=[col], drop_first=True)
            self.logger_object.log(
                'encoding for categorical values successful. Exited the encode_categorical_columns method of the Preprocessor class')
            return self.cat_df

        except Exception as e:
            pre_processor_exception = PreprocessorException(
                "Failed during set in module [{0}] class [{1}] method [{2}]"
                    .format(self.__module__, Preprocessor.__name__,
                            self.encode_categorical_columns_income_prediction.__name__))

    def encode_categorical_values_prediction(self, data):
        """
                                               Method Name: encodeCategoricalValuesPrediction
                                               Description: This method encodes all the categorical values in the prediction set.
                                               Output: A Dataframe which has all the categorical values encoded.
                                               On Failure: Raise Exception

                                               Written By: iNeuron Intelligence
                                               Version: 1.0
                                               Revisions: None
                            """

        # We can map the categorical values like below:
        try:
            data['sex'] = data['sex'].map({'F': 0, 'M': 1})
            cat_data = data.drop(['age', 'T3', 'TT4', 'T4U', 'FTI', 'sex'],
                                 axis=1)  # we do not want to encode values with int or float type
            # except for 'Sex' column all the other columns with two categorical data have same value 'f' and 't'.
            # so instead of mapping indvidually, let's do a smarter work
            for column in cat_data.columns:
                if (data[column].nunique()) == 1:
                    if data[column].unique()[0] == 'f' or data[column].unique()[
                        0] == 'F':  # map the variables same as we did in training i.e. if only 'f' comes map as 0 as done in training
                        data[column] = data[column].map({data[column].unique()[0]: 0})
                    else:
                        data[column] = data[column].map({data[column].unique()[0]: 1})
                elif (data[column].nunique()) == 2:
                    data[column] = data[column].map({'f': 0, 't': 1})

            # we will use get dummies for 'referral_source'
            data = pd.get_dummies(data, columns=['referral_source'])
            return data
        except Exception as e:
            pre_processor_exception = PreprocessorException(
                "Failed during method encodes all the categorical values in the training set in module [{0}] class [{1}] method [{2}]"
                    .format(self.__module__, Preprocessor.__name__,
                            self.encode_categorical_values_prediction.__name__))
            raise Exception(pre_processor_exception.error_message_detail(str(e), sys)) from e

    def encode_categorical_values_back_order_prediction(self, data):
        """
        Method Name: encodeCategoricalValues
        Description: This method encodes all the categorical values in the training set.
        Output: A Dataframe which has all the categorical values encoded.
        On Failure: Raise Exception

        Written By: iNeuron Intelligence
        Version: 1.0
        Revisions: None
        """
        try:
            str_col = ["potential_issue", "deck_risk", "ppap_risk", "stop_auto_buy", "rev_stop"]
            for column in str_col:
                data[column] = data[column].map({"Yes": 0, "No": 1})

            return data
        except Exception as e:
            pre_processor_exception = PreprocessorException(
                "Failed during method encodes all the categorical values in "
                "the training set in module [{0}] class [{1}] method [{2}]"
                    .format(self.__module__, Preprocessor.__name__,
                            self.encode_categorical_values_back_order_prediction.__name__))
            raise Exception(pre_processor_exception.error_message_detail(str(e), sys)) from e

    def encode_categorical_values_back_order(self, data):
        """
                                        Method Name: encodeCategoricalValues
                                        Description: This method encodes all the categorical values in the training set.
                                        Output: A Dataframe which has all the categorical values encoded.
                                        On Failure: Raise Exception

                                        Written By: iNeuron Intelligence
                                        Version: 1.0
                                        Revisions: None
                     """
        try:
            str_col = ["potential_issue", "deck_risk", "oe_constraint", "ppap_risk", "stop_auto_buy", "rev_stop",
                       "went_on_backorder"]
            for column in str_col:
                data[column] = data[column].map({"Yes": 0, "No": 1})
            return data
        except Exception as e:
            pre_processor_exception = PreprocessorException(
                "Failed during method encodes all the categorical values in the training set in module [{0}] class [{1}] method [{2}]"
                    .format(self.__module__, Preprocessor.__name__,
                            self.encode_categorical_values_back_order.__name__))
            raise Exception(pre_processor_exception.error_message_detail(str(e), sys)) from e

    def log_transform(self, X):
        """

        :param X: Accept X as pandas df and calculated
        :return: dataframe after applying log on each column
        """
        try:
            for column in X.columns:
                X[column] += 1
                X[column] = np.log(X[column])

            return X
        except Exception as e:
            pre_processor_exception = PreprocessorException(
                "Failed in module [{0}] class [{1}] method [{2}]"
                    .format(self.__module__, Preprocessor.__name__,
                            self.log_transform.__name__))
            raise Exception(pre_processor_exception.error_message_detail(str(e), sys)) from e

    def separate_label_feature(self, data, label_column_name):
        """
                        Method Name: separate_label_feature
                        Description: This method separates the features and a Label Coulmns.
                        Output: Returns two separate Dataframes, one containing features and the other containing Labels .
                        On Failure: Raise Exception

                        Written By: iNeuron Intelligence
                        Version: 1.0
                        Revisions: None

                """
        try:
            self.logger_object.log('Entered the separate_label_feature method of the Preprocessor class')

            x = data.drop(labels=label_column_name, axis=1)  # drop the columns specified and separate the feature
            # columns
            y = data[label_column_name]  # Filter the Label columns
            self.logger_object.log('Label Separation Successful. Exited the separate_label_feature method of the '
                                   'Preprocessor class')

            return x, y
        except Exception as e:
            pre_processor_exception = PreprocessorException(
                "Failed during seperating label  in module [{0}] class [{1}] method [{2}]"
                    .format(self.__module__, Preprocessor.__name__,
                            self.separate_label_feature.__name__))
            raise Exception(pre_processor_exception.error_message_detail(str(e), sys)) from e

    def plot_bar_plot(self, data, target_column_name):
        """

        :param data: dataframe
        :param target_column_name: target column
        :return: notihing
        """
        try:
            bar_plot_label = data[target_column_name].value_counts().index.to_list()
            bar_plot_data = data[target_column_name].value_counts().to_list()
            AccurayGraph().save_accuracy_bar_graph(model_name_list=bar_plot_label, accuracy_score_list=bar_plot_data,
                                                   project_id=self.project_id,
                                                   execution_id=self.logger_object.execution_id,
                                                   file_object=self.file_object,
                                                   title="Target Attribute Distribution",
                                                   x_label="Type of Class ",
                                                   y_label="Number of record in each class"
                                                   )
        except Exception as e:
            pre_processor_exception = PreprocessorException(
                "Failed  in module [{0}] class [{1}] method [{2}]"
                    .format(self.__module__, Preprocessor.__name__,
                            self.plot_bar_plot.__name__))
            raise Exception(pre_processor_exception.error_message_detail(str(e), sys)) from e

    def plot_pie_plot(self, data, target_column_name):
        """

        :param data: dataframe
        :param target_column_name: target column
        :return: notihing
        """
        try:
            plot_label = data[target_column_name].value_counts().index.to_list()
            plot_data = data[target_column_name].value_counts().to_list()
            AccurayGraph().save_pie_plot(data=plot_data, label=plot_label, project_id=self.project_id,
                                         execution_id=self.logger_object.execution_id, file_object=self.file_object,
                                         title="Target attribute distribution"
                                         )

        except Exception as e:
            pre_processor_exception = PreprocessorException(
                "Failed  in module [{0}] class [{1}] method [{2}]"
                    .format(self.__module__, Preprocessor.__name__,
                            self.plot_pie_plot.__name__))
            raise Exception(pre_processor_exception.error_message_detail(str(e), sys)) from e

    def encode_categorical_values_mushroom(self, data):
        """
        Method Name: encodeCategoricalValues
        Description: This method encodes all the categorical values in the training set.
        Output: A Dataframe which has all the categorical values encoded.
        On Failure: Raise Exception

        Written By: iNeuron Intelligence
        Version: 1.0
        Revisions: None
        """
        try:

            data["class"] = data["class"].map({"'p'": 1, "'e'": 2})

            for column in data.drop(['class'], axis=1).columns:
                data = pd.get_dummies(data, columns=[column])

            mushroom_column_store_file_path = self.initializer.get_encoded_column_name_file_path(self.project_id)
            column_name_data_file = self.initializer.get_encoded_column_file_name()
            self.file_object.write_file_content(mushroom_column_store_file_path, column_name_data_file,
                                                data.drop(['class'], axis=1).columns.values)

            return data
        except Exception as e:
            pre_processor_exception = PreprocessorException(
                "Failed  in module [{0}] class [{1}] method [{2}]"
                    .format(self.__module__, Preprocessor.__name__,
                            self.encode_categorical_values_mushroom.__name__))
            raise Exception(pre_processor_exception.error_message_detail(str(e), sys)) from e

    def encode_categorical_values_scania_truck(self, data):
        """
        Method Name: encodeCategoricalValues
        Description: This method encodes all the categorical values in the training set.
        Output: A Dataframe which has all the categorical values encoded.
        On Failure: Raise Exception

        Written By: iNeuron Intelligence
        Version: 1.0
        Revisions: None
        """
        try:
            self.logger_object.log("Started encoding categorical variable")
            data['class'] = data['class'].map({'neg': 0, 'pos': 1})
            self.logger_object.log("Completed encoding categorical variable")
            return data
        except Exception as e:
            pre_processor_exception = PreprocessorException(
                "Failed  in module [{0}] class [{1}] method [{2}]"
                    .format(self.__module__, Preprocessor.__name__,
                            self.encode_categorical_values_scania_truck.__name__))
            raise Exception(pre_processor_exception.error_message_detail(str(e), sys)) from e

    def encode_categorical_columns_fraud_detection(self, data):
        """
                                                Method Name: encode_categorical_columns
                                                Description: This method encodes the categorical values to numeric values.
                                                Output: dataframe with categorical values converted to numerical values
                                                On Failure: Raise Exception

                                                Written By: iNeuron Intelligence
                                                Version: 1.0
                                                Revisions: None
                             """

        try:
            self.logger_object.log('Entered the encode_categorical_columns method of the Preprocessor class')
            self.data = data
            self.cat_df = self.data.select_dtypes(include=['object']).copy()
            self.cat_df['policy_csl'] = self.cat_df['policy_csl'].map({'100/300': 1, '250/500': 2.5, '500/1000': 5})
            self.cat_df['insured_education_level'] = self.cat_df['insured_education_level'].map(
                {'JD': 1, 'High School': 2, 'College': 3, 'Masters': 4, 'Associate': 5, 'MD': 6, 'PhD': 7})
            self.cat_df['incident_severity'] = self.cat_df['incident_severity'].map(
                {'Trivial Damage': 1, 'Minor Damage': 2, 'Major Damage': 3, 'Total Loss': 4})
            self.cat_df['insured_sex'] = self.cat_df['insured_sex'].map({'FEMALE': 0, 'MALE': 1})
            self.cat_df['property_damage'] = self.cat_df['property_damage'].map({'NO': 0, 'YES': 1})
            self.cat_df['police_report_available'] = self.cat_df['police_report_available'].map({'NO': 0, 'YES': 1})
            try:
                # code block for training
                self.cat_df['fraud_reported'] = self.cat_df['fraud_reported'].map({'N': 0, 'Y': 1})
                self.cols_to_drop = ['policy_csl', 'insured_education_level', 'incident_severity', 'insured_sex',
                                     'property_damage', 'police_report_available', 'fraud_reported']
            except:
                # code block for Prediction
                self.cols_to_drop = ['policy_csl', 'insured_education_level', 'incident_severity', 'insured_sex',
                                     'property_damage', 'police_report_available']
            # Using the dummy encoding to encode the categorical columns to numerical ones

            for col in self.cat_df.drop(columns=self.cols_to_drop).columns:
                self.cat_df = pd.get_dummies(self.cat_df, columns=[col], prefix=[col], drop_first=True)

            self.data.drop(columns=self.data.select_dtypes(include=['object']).columns, inplace=True)
            self.data = pd.concat([self.cat_df, self.data], axis=1)
            self.logger_object.log(
                'encoding for categorical values successful. Exited the encode_categorical_columns method of the Preprocessor class')
            return self.data
        except Exception as e:
            pre_processor_exception = PreprocessorException(
                "Failed  in module [{0}] class [{1}] method [{2}]"
                    .format(self.__module__, Preprocessor.__name__,
                            self.encode_categorical_columns_fraud_detection.__name__))
            raise Exception(pre_processor_exception.error_message_detail(str(e), sys)) from e

    def pca_transformation_scania_truck(self, X_scaled_data):
        try:
            pca = PCA(n_components=100)
            new_data = pca.fit_transform(X_scaled_data)
            principal_x = pd.DataFrame(new_data, index=self.data.index)
            return principal_x
        except Exception as e:
            pre_processor_exception = PreprocessorException(
                "Failed  in module [{0}] class [{1}] method [{2}]"
                    .format(self.__module__, Preprocessor.__name__,
                            self.pca_transformation_scania_truck.__name__))
            raise Exception(pre_processor_exception.error_message_detail(str(e), sys)) from e

    def pca_transform_back_order(self, X_scaled_data):
        try:

            self.data = X_scaled_data
            self.num_df = self.data.drop(["potential_issue", "deck_risk", "ppap_risk", "stop_auto_buy", "rev_stop"],
                                         axis=1)
            pca = PCA(n_components=7)
            new_data = pca.fit_transform(self.num_df)
            principal_x = pd.DataFrame(new_data,
                                       columns=['PC-1', 'PC-2', 'PC-3', 'PC-4', 'PC-5', 'PC-6', 'PC-7'],
                                       index=self.num_df.index)
            self.data.drop(columns=self.num_df.columns, inplace=True)
            self.data = pd.concat([principal_x, self.data], axis=1)
            return self.data
        except Exception as e:
            pre_processor_exception = PreprocessorException(
                "Failed  in module [{0}] class [{1}] method [{2}]"
                    .format(self.__module__, Preprocessor.__name__,
                            self.pca_transform_back_order.__name__))
        raise Exception(pre_processor_exception.error_message_detail(str(e), sys)) from e

    def scale_numerical_columns_back_order(self, data):
        """
                                                        Method Name: scale_numerical_columns
                                                        Description: This method scales the numerical values using the Standard scaler.
                                                        Output: A dataframe with scaled values
                                                        On Failure: Raise Exception

                                                        Written By: iNeuron Intelligence
                                                        Version: 1.0
                                                        Revisions: None
                                     """

        try:
            self.logger_object.log('Entered the scale_numerical_columns method of the Preprocessor class')
            self.data = data
            self.num_df = self.data.drop(["potential_issue", "deck_risk", "ppap_risk", "stop_auto_buy", "rev_stop"],
                                         axis=1)
            self.scaler = StandardScaler()
            self.scaled_data = self.scaler.fit_transform(self.num_df)
            self.scaled_num_df = pd.DataFrame(data=self.scaled_data, columns=self.num_df.columns, index=self.data.index)
            self.data.drop(columns=self.scaled_num_df.columns, inplace=True)
            self.data = pd.concat([self.scaled_num_df, self.data], axis=1)
            self.logger_object.log(
                'scaling for numerical values successful. Exited the scale_numerical_columns method of the Preprocessor class')
            return self.data
        except Exception as e:
            pre_processor_exception = PreprocessorException(
                "Failed  in module [{0}] class [{1}] method [{2}]"
                    .format(self.__module__, Preprocessor.__name__,
                            self.scale_numerical_columns_back_order.__name__))
            raise Exception(pre_processor_exception.error_message_detail(str(e), sys)) from e

    def scale_numerical_columns_scania_truck(self, data):
        """
                                                        Method Name: scale_numerical_columns
                                                        Description: This method scales the numerical values using the Standard scaler.
                                                        Output: A dataframe with scaled values
                                                        On Failure: Raise Exception

                                                        Written By: iNeuron Intelligence
                                                        Version: 1.0
                                                        Revisions: None
                                     """

        try:
            self.logger_object.log('Entered the scale_numerical_columns method of the Preprocessor class')
            self.data = data
            # self.num_df = self.data.drop(["potential_issue","deck_risk","ppap_risk","stop_auto_buy","rev_stop"],axis=1)
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(data)
            scaled_num_df = pd.DataFrame(data=scaled_data, columns=data.columns, index=data.index)

            self.logger_object.log(
                'scaling for numerical values successful. Exited the scale_numerical_columns method of the Preprocessor class')
            return scaled_num_df
        except Exception as e:
            pre_processor_exception = PreprocessorException(
                "Failed  in module [{0}] class [{1}] method [{2}]"
                    .format(self.__module__, Preprocessor.__name__,
                            self.scale_numerical_columns_scania_truck.__name__))
            raise Exception(pre_processor_exception.error_message_detail(str(e), sys)) from e

    def scale_numerical_columns_bigmart_sales(self, data):
        """
                                                        Method Name: scale_numerical_columns
                                                        Description: This method scales the numerical values using the Standard scaler.
                                                        Output: A dataframe with scaled values
                                                        On Failure: Raise Exception

                                                        Written By: iNeuron Intelligence
                                                        Version: 1.0
                                                        Revisions: None
                                     """
        try:
            self.logger_object.log('Entered the scale_numerical_columns method of the Preprocessor class')

            self.data = data
            self.num_df = self.data[['Item_Weight', 'Item_Visibility', 'Item_MRP', 'Outlet_Years']]
            self.scaler = StandardScaler()
            self.scaled_data = self.scaler.fit_transform(self.num_df)
            self.scaled_num_df = pd.DataFrame(data=self.scaled_data, columns=self.num_df.columns, index=self.data.index)
            self.data.drop(columns=self.scaled_num_df.columns, inplace=True)
            self.data = pd.concat([self.scaled_num_df, self.data], axis=1)
            self.logger_object.log(
                'scaling for numerical values successful. Exited the scale_numerical_columns method of the Preprocessor class')
            return self.data
        except Exception as e:
            pre_processor_exception = PreprocessorException(
                "Failed  in module [{0}] class [{1}] method [{2}]"
                    .format(self.__module__, Preprocessor.__name__,
                            self.scale_numerical_columns_bigmart_sales.__name__))
            raise Exception(pre_processor_exception.error_message_detail(str(e), sys)) from e

    def standard_scaling_data_of_column(self, X):
        """

        :param X: Accept df
        :return: apply standard scaling on each column and returned scaled column
        """
        try:
            scalar = StandardScaler()
            X_scaled = scalar.fit_transform(X)
            return X_scaled
        except Exception as e:
            pre_processor_exception = PreprocessorException(
                "Failed  in module [{0}] class [{1}] method [{2}]"
                    .format(self.__module__, Preprocessor.__name__,
                            self.standard_scaling_data_of_column.__name__))
            raise Exception(pre_processor_exception.error_message_detail(str(e), sys)) from e

    def scale_numerical_columns_credit_default_or_income_prediction(self, data):
        """
                                                        Method Name: scale_numerical_columns
                                                        Description: This method scales the numerical values using the Standard scaler.
                                                        Output: A dataframe with scaled
                                                        On Failure: Raise Exception

                                                        Written By: iNeuron Intelligence
                                                        Version: 1.0
                                                        Revisions: None
                                     """

        try:
            self.logger_object.log('Entered the scale_numerical_columns method of the Preprocessor class')
            self.data = data
            self.num_df = self.data.select_dtypes(include=['int64']).copy()
            self.scaler = StandardScaler()
            self.scaled_data = self.scaler.fit_transform(self.num_df)
            self.scaled_num_df = pd.DataFrame(data=self.scaled_data, columns=self.num_df.columns)
            self.logger_object.log(
                'scaling for numerical values successful. Exited the scale_numerical_columns method of the Preprocessor class')
            return self.scaled_num_df
        except Exception as e:
            pre_processor_exception = PreprocessorException(
                "Failed  in module [{0}] class [{1}] method [{2}]"
                    .format(self.__module__, Preprocessor.__name__,
                            self.scale_numerical_columns_credit_default_or_income_prediction.__name__))
            raise Exception(pre_processor_exception.error_message_detail(str(e), sys)) from e

    def scale_numerical_columns_fraud_detection(self, data):
        """
        Method Name: scale_numerical_columns
        Description: This method scales the numerical values using the Standard scaler.
        Output: A dataframe with scaled values
        On Failure: Raise Exception

        Written By: iNeuron Intelligence
        Version: 1.0
        Revisions: None
        """

        try:
            self.logger_object.log('Entered the scale_numerical_columns method of the Preprocessor class')
            self.data = data
            self.num_df = self.data[['months_as_customer', 'policy_deductable', 'umbrella_limit',
                                     'capital-gains', 'capital-loss', 'incident_hour_of_the_day',
                                     'number_of_vehicles_involved', 'bodily_injuries', 'witnesses', 'injury_claim',
                                     'property_claim',
                                     'vehicle_claim']]
            self.scaler = StandardScaler()
            self.scaled_data = self.scaler.fit_transform(self.num_df)
            self.scaled_num_df = pd.DataFrame(data=self.scaled_data, columns=self.num_df.columns, index=self.data.index)
            self.data.drop(columns=self.scaled_num_df.columns, inplace=True)
            self.data = pd.concat([self.scaled_num_df, self.data], axis=1)

            self.logger_object.log(
                'scaling for numerical values successful. Exited the scale_numerical_columns method of the Preprocessor class')
            return self.data
        except Exception as e:
            pre_processor_exception = PreprocessorException(
                "Failed  in module [{0}] class [{1}] method [{2}]"
                    .format(self.__module__, Preprocessor.__name__,
                            self.scale_numerical_columns_fraud_detection.__name__))
            raise Exception(pre_processor_exception.error_message_detail(str(e), sys)) from e

    def handle_missing_values_scania_truck(self, data):
        try:
            data = data[data.columns[data.isnull().mean() < 0.6]]
            data = data.apply(pd.to_numeric)
            for col in data.columns:
                data[col] = data[col].replace(np.NaN, data[col].mean())
            return data
        except Exception as e:
            pre_processor_exception = PreprocessorException(
                "Failed  in module [{0}] class [{1}] method [{2}]"
                    .format(self.__module__, Preprocessor.__name__,
                            self.handle_missing_values_scania_truck.__name__))
            raise Exception(pre_processor_exception.error_message_detail(str(e), sys)) from e

    def impute_missing_values_mushroom(self, data, cols_with_missing_values):
        """
                                        Method Name: impute_missing_values
                                        Description: This method replaces all the missing values in the Dataframe using KNN Imputer.
                                        Output: A Dataframe which has all the missing values imputed.
                                        On Failure: Raise Exception

                                        Written By: iNeuron Intelligence
                                        Version: 1.0
                                        Revisions: None
                     """

        try:
            self.logger_object.log('Entered the impute_missing_values method of the Preprocessor class')
            data = data
            cols_with_missing_values = cols_with_missing_values
            imputer = CategoricalImputer()
            for col in cols_with_missing_values:
                try:
                    data[col] = imputer.fit_transform(data[col])
                except ValueError as e:
                    column_value = list(data[col].unique())
                    if np.nan in column_value:
                        column_value.remove(np.nan)
                        data[col] = data[[col]].replace(np.nan, column_value[0])

            self.logger_object.log(
                'Imputing missing values Successful. Exited the impute_missing_values method of the Preprocessor class')
            return data

        except Exception as e:
            pre_processor_exception = PreprocessorException(
                "Failed during imputing missing value in mushroom label  in module [{0}] class [{1}] method [{2}]"
                    .format(self.__module__, Preprocessor.__name__,
                            self.impute_missing_values_mushroom.__name__))
            raise Exception(pre_processor_exception.error_message_detail(str(e), sys)) from e

    def convert_cost_to_number(self, data):
        """
                        Method Name: convertCostToNumber
                        Description: This method converts the cost column to floating point numbers

                        Written By: iNeuron Intelligence
                        Version: 1.0
                        Revisions: None

                                """
        try:
            self.logger_object.log("Started converting cost to number.")
            data['approx_cost(for two people)'] = data['approx_cost(for two people)'].astype(
                str)  # Changing the cost to string
            data['approx_cost(for two people)'] = data['approx_cost(for two people)'].apply(
                lambda x: x.replace(',', '.'))  # Using lambda function to replace ',' from cost
            data['approx_cost(for two people)'] = data['approx_cost(for two people)'].astype(
                float)  # Changing the cost to Float
            self.logger_object.log("Completed the process of converting cost to number.")
            return data
        except Exception as e:
            pre_processor_exception = PreprocessorException(
                "Failed during imputing missing value in mushroom label  in module [{0}] class [{1}] method [{2}]"
                    .format(self.__module__, Preprocessor.__name__,
                            self.convert_cost_to_number.__name__))
            raise Exception(pre_processor_exception.error_message_detail(str(e), sys)) from e

    def encode_categorical_values_zomato(self, data):
        """
        Method Name: encodeCategoricalValues
        Description: This method encodes all the categorical values in the training set.
        Output: A Dataframe which has all the categorical values encoded.
        On Failure: Raise Exception

        Written By: iNeuron Intelligence
        Version: 1.0
        Revisions: None
        """
        try:
            data["online_order"] = data["online_order"].map({'Yes': 1, 'No': 0})
            data["book_table"] = data["book_table"].map({'Yes': 1, 'No': 0})
            for column in data.columns[
                ~data.columns.isin(['book_table', 'online_order', 'rate', 'approx_cost(for two people)', 'votes'])]:
                data[column] = data[column].factorize()[0]
            return data
        except Exception as e:
            pre_processor_exception = PreprocessorException(
                "Failed during imputing missing value in mushroom label  in module [{0}] class [{1}] method [{2}]"
                    .format(self.__module__, Preprocessor.__name__,
                            self.encode_categorical_values_zomato.__name__))
            raise Exception(pre_processor_exception.error_message_detail(str(e), sys)) from e

    def is_null_present_in_columns(self, data):
        """
                                Method Name: is_null_present
                                Description: This method checks whether there are null values present in the pandas Dataframe or not.
                                Output: Returns True if null values are present in the DataFrame, False if they are not present and
                                        returns the list of columns for which null values are present.
                                On Failure: Raise Exception

                                Written By: iNeuron Intelligence
                                Version: 1.0
                                Revisions: None

                        """
        self.logger_object.log('Entered the is_null_present method of the Preprocessor class')
        self.null_present = False
        self.cols_with_missing_values = []
        self.cols = data.columns
        try:
            self.null_counts = data.isna().sum()  # check for the count of null values per column
            for i in range(len(self.null_counts)):
                if self.null_counts[i] > 0:
                    self.null_present = True
                    self.cols_with_missing_values.append(self.cols[i])
            if (self.null_present):  # write the logs to see which columns have null values
                self.dataframe_with_null = pd.DataFrame()
                self.dataframe_with_null['columns'] = data.columns
                self.dataframe_with_null['missing values count'] = np.asarray(data.isna().sum())
                preprocessing_data_path = self.initializer.get_training_preprocessing_data_path(self.project_id)
                null_value_file_name = self.initializer.get_null_value_csv_file_name()
                self.file_object.write_file_content(preprocessing_data_path, null_value_file_name,
                                                    self.dataframe_with_null, over_write=True)

                # self.dataframe_with_null.to_csv('preprocessing_data/null_values.csv') # storing the null column information to file
            self.logger_object.log(
                'Finding missing values is a success.Data written to the null values file. Exited the is_null_present method of the Preprocessor class')
            return self.null_present, self.cols_with_missing_values
        except Exception as e:
            pre_processor_exception = PreprocessorException(
                "Exception occurred in is_null_present method of the Preprocessor class. Exception in module [{0}] "
                "class [{1}] method [{2}] "
                    .format(self.__module__, Preprocessor.__name__,
                            self.is_null_present_in_columns.__name__))
            raise Exception(pre_processor_exception.error_message_detail(str(e), sys)) from e

    def is_null_present(self, data):
        """
                                Method Name: is_null_present
                                Description: This method checks whether there are null values present in the pandas Dataframe or not.
                                Output: Returns a Boolean Value. True if null values are present in the DataFrame, False if they are not present.
                                On Failure: Raise Exception

                                Written By: iNeuron Intelligence
                                Version: 1.0
                                Revisions: None

                        """

        try:
            self.logger_object.log('Entered the is_null_present method of the Preprocessor class')
            null_present = False
            null_counts = data.isna().sum()  # check for the count of null values per column
            for i in null_counts:
                if i > 0:
                    null_present = True
                    break
            if null_present:  # write the logs to see which columns have null values
                dataframe_with_null = pd.DataFrame()
                dataframe_with_null['columns'] = data.columns
                dataframe_with_null['missing values count'] = np.asarray(data.isna().sum())
                preprocessing_data_path = self.initializer.get_training_preprocessing_data_path(self.project_id)
                null_value_file_name = self.initializer.get_null_value_csv_file_name()
                self.file_object.write_file_content(preprocessing_data_path, null_value_file_name, dataframe_with_null,
                                                    over_write=True)
                # dataframe_with_null.to_csv(
                #    'preprocessing_data/null_values.csv')  # storing the null column information to file
            self.logger_object.log('Finding missing values is a success.Data written to the null values file. Exited '
                                   'the is_null_present method of the Preprocessor class')
            return null_present
        except Exception as e:
            pre_processor_exception = PreprocessorException(
                "Exception occurred in is_null_present method of the Preprocessor class. Exception in module [{0}] "
                "class [{1}] method [{2}] "
                    .format(self.__module__, Preprocessor.__name__,
                            self.remove_columns.__name__))
            raise Exception(pre_processor_exception.error_message_detail(str(e), sys)) from e

    def impute_missing_values(self, data):
        """
                                        Method Name: impute_missing_values
                                        Description: This method replaces all the missing values in the Dataframe using KNN Imputer.
                                        Output: A Dataframe which has all the missing values imputed.
                                        On Failure: Raise Exception

                                        Written By: iNeuron Intelligence
                                        Version: 1.0
                                        Revisions: None
                     """
        try:
            self.logger_object.log('Entered the impute_missing_values method of the Preprocessor class')
            imputer = KNNImputer(n_neighbors=3, weights='uniform', missing_values=np.nan)
            new_array = imputer.fit_transform(data)  # impute the missing values
            # convert the nd-array returned in the step above to a Dataframe
            new_data = pd.DataFrame(data=new_array, columns=data.columns)
            self.logger_object.log('Imputing missing values Successful. Exited the impute_missing_values method of '
                                   'the Preprocessor class')

            # new_data=data.fillna(data.mean())
            return new_data
        except Exception as e:
            pre_processor_exception = PreprocessorException(
                "Exception occurred in impute_missing_values method of the Preprocessor class. in module [{0}] "
                "class [{1}] method [{2}] "
                    .format(self.__module__, Preprocessor.__name__,
                            self.impute_missing_values.__name__))
            raise Exception(pre_processor_exception.error_message_detail(str(e), sys)) from e

    def get_columns_with_zero_std_deviation(self, data):
        """
                                                Method Name: get_columns_with_zero_std_deviation
                                                Description: This method finds out the columns which have a standard deviation of zero.
                                                Output: List of the columns with standard deviation of zero
                                                On Failure: Raise Exception

                                                Written By: iNeuron Intelligence
                                                Version: 1.0
                                                Revisions: None
                             """

        try:
            self.logger_object.log('Entered the get_columns_with_zero_std_deviation method of the Preprocessor class')
            columns = data.columns
            data_n = data.describe()
            col_to_drop = []
            for x in columns:
                if data_n[x]['std'] == 0:  # check if standard deviation is zero
                    col_to_drop.append(x)  # prepare the list of columns with standard deviation zero
            self.logger_object.log('Column search for Standard Deviation of Zero Successful. Exited the '
                                   'get_columns_with_zero_std_deviation method of the Preprocessor class')
            return col_to_drop
        except Exception as e:
            pre_processor_exception = PreprocessorException(
                "Column search for Standard Deviation of Zero Failed in module [{0}] "
                "class [{1}] method [{2}] "
                    .format(self.__module__, Preprocessor.__name__,
                            self.get_columns_with_zero_std_deviation.__name__))
            raise Exception(pre_processor_exception.error_message_detail(str(e), sys)) from e
