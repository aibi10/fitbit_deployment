import uuid

import numpy

import pandas
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBClassifier, XGBRegressor
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import roc_auc_score, accuracy_score, r2_score, roc_curve
import sys
from exception_layer.generic_exception.generic_exception import GenericException as RandomForestClassificationException
from exception_layer.generic_exception.generic_exception import GenericException as XGBoostClassificationException
from exception_layer.generic_exception.generic_exception import GenericException as ModelFinderException
from plotly_dash.accuracy_graph.accuracy_graph import AccurayGraph
from sklearn.naive_bayes import GaussianNB
from project_library_layer.initializer.initializer import Initializer
from sklearn.linear_model import Ridge, Lasso, RidgeCV, LassoCV, ElasticNet, ElasticNetCV


class ModelFinder:
    """
                This class shall  be used to find the model with best accuracy and AUC score.
                Written By: iNeuron Intelligence
                Version: 1.0
                Revisions: None

                """

    def __init__(self, project_id, file_object, logger_object):
        try:
            self.project_id = project_id
            self.file_object = file_object
            self.logger_object = logger_object
            self.clf = RandomForestClassifier()
            self.knn = KNeighborsClassifier()
            self.xgb = XGBClassifier(objective='binary:logistic')
            self.sv_classifier = SVC()
            self.gnb = GaussianNB()
            self.linearReg = LinearRegression()
            self.RandomForestReg = RandomForestRegressor()
            self.DecisionTreeReg = DecisionTreeRegressor()
            self.sv_regressor = SVR()
            self.sgd_regression = SGDRegressor()
            self.initailizer = Initializer()
            self.model_name = []
            self.model = []
            self.score = []

        except Exception as e:
            model_finder = ModelFinderException(
                "Failed during object instantiation in module [{0}] class [{1}] method [{2}]"
                    .format(self.__module__, ModelFinder.__name__,
                            self.__init__.__name__))
            raise Exception(model_finder.error_message_detail(str(e), sys)) from e

    def get_best_params_for_ridge_regression(self, train_x, train_y):
        try:
            self.logger_object.log("Entered the get best params for Ridge Repressor")
            alphas = numpy.random.uniform(low=0, high=10, size=(50,))
            ridge_cv = RidgeCV(alphas=alphas, cv=5, normalize=True)
            ridge_cv.fit(train_x, train_y)
            alpha = ridge_cv.alpha_
            ridge_model = Ridge(alpha=alpha)
            ridge_model.fit(train_x, train_y)
            self.logger_object.log(
                'Ridge Regressor best params <alpha value: ' + str(ridge_cv.alpha_) + '>. Exited the '
                'get_best_params_for_ridge_regression method of the Model_Finder class')
            return ridge_model
        except Exception as e:
            model_finder = ModelFinderException(
                "Model Selection Failed in module [{0}] class [{1}] method [{2}]"
                    .format(self.__module__, ModelFinder.__name__,
                            self.get_best_params_for_ridge_regression.__name__))
            raise Exception(model_finder.error_message_detail(str(e), sys)) from e

    def get_best_params_for_support_vector_regressor(self, train_x, train_y):
        try:
            self.logger_object.log("Entered the get best params for Support Vector Repressor")

            param_grid = {'C': [0.1, 1, 10, 50, 100, 500], 'gamma': [1, 0.5, 0.1, 0.01, 0.001]}

            grid = GridSearchCV(SVR(), param_grid, verbose=3, cv=5)

            grid.fit(train_x, train_y)

            C = grid.best_params_['C']
            gamma = grid.best_params_['gamma']
            svr_reg = SVR(C=C, gamma=gamma)
            svr_reg.fit(train_x, train_y)

            self.logger_object.log('Support Vector Regressor best params: ' + str(grid.best_params_) + '. Exited the '
                                   'get_best_params_for_support_vector_regressor method of the Model_Finder class')
            return svr_reg
        except Exception as e:
            model_finder = ModelFinderException(
                "Model Selection Failed in module [{0}] class [{1}] method [{2}]"
                    .format(self.__module__, ModelFinder.__name__,
                            self.get_best_params_for_support_vector_regressor.__name__))
            raise Exception(model_finder.error_message_detail(str(e), sys)) from e

    def get_best_params_for_random_forest(self, train_x, train_y):
        """
                                Method Name: get_best_params_for_random_forest
                                Description: get the parameters for Random Forest Algorithm which give the best accuracy.
                                             Use Hyper Parameter Tuning.
                                Output: The model with the best parameters
                                On Failure: Raise Exception

                                Written By: iNeuron Intelligence
                                Version: 1.0
                                Revisions: None

                        """
        try:
            self.logger_object.log('Entered the get_best_params_for_random_forest method of the Model_Finder class')
            # initializing with different combination of parameters
            param_grid = {"n_estimators": [10, 130], "criterion": ['gini', 'entropy'],
                          "max_depth": range(2, 4, 1), "max_features": ['auto', 'log2']}
            # Creating an object of the Grid Search class
            grid = GridSearchCV(estimator=self.clf, param_grid=param_grid, cv=5, verbose=3)
            # finding the best parameters
            grid.fit(train_x, train_y)

            # extracting the best parameters
            criterion = grid.best_params_['criterion']
            max_depth = grid.best_params_['max_depth']
            max_features = grid.best_params_['max_features']
            n_estimators = grid.best_params_['n_estimators']

            # creating a new model with the best parameters
            self.clf = RandomForestClassifier(n_estimators=n_estimators, criterion=criterion,
                                              max_depth=max_depth, max_features=max_features)
            # training the mew model
            self.clf.fit(train_x, train_y)
            self.logger_object.log('Random Forest best params: ' + str(grid.best_params_) + '. Exited the '
                                   'get_best_params_for_random_forest method of the Model_Finder class')

            return self.clf
        except Exception as e:
            random_clf_exception = RandomForestClassificationException(
                "Random Forest Parameter tuning  failed in module [{0}] class [{1}] method [{2}]"
                    .format(self.__module__, ModelFinder.__name__,
                            self.get_best_params_for_random_forest.__name__))
            raise Exception(random_clf_exception.error_message_detail(str(e), sys)) from e

    def get_best_params_for_xgboost(self, train_x, train_y):

        """
        Method Name: get_best_params_for_xgboost
        Description: get the parameters for XGBoost Algorithm which give the best accuracy.
        Use Hyper Parameter Tuning.
        Output: The model with the best parameters
        On Failure: Raise Exception

        Written By: iNeuron Intelligence
        Version: 1.0
        Revisions: None

        """

        try:
            self.logger_object.log('Entered the get_best_params_for_xgboost method of the Model_Finder class')
            # initializing with different combination of parameters
            param_grid_xgboost = {

                'learning_rate': [0.5, 0.001],
                'max_depth': [20],

                'n_estimators': [10, 200]

            }
            # Creating an object of the Grid Search class
            grid = GridSearchCV(XGBClassifier(objective='binary:logistic'), param_grid_xgboost, verbose=3, cv=5)
            # finding the best parameters
            grid.fit(train_x, train_y)

            # extracting the best parameters
            learning_rate = grid.best_params_['learning_rate']
            max_depth = grid.best_params_['max_depth']
            n_estimators = grid.best_params_['n_estimators']

            # creating a new model with the best parameters
            self.xgb = XGBClassifier(learning_rate=learning_rate, max_depth=max_depth, n_estimators=n_estimators)
            # training the mew model
            self.xgb.fit(train_x, train_y)
            self.logger_object.log('XGBoost best params: ' + str(grid.best_params_) + '. Exited the '
                                                                                      'get_best_params_for_xgboost method of the Model_Finder class')
            return self.xgb
        except Exception as e:
            xg_boost_clf_exception = XGBoostClassificationException(
                "XGBoost Parameter tuning  failed in module [{0}] class [{1}] method [{2}]"
                    .format(self.__module__, ModelFinder.__name__,
                            self.get_best_params_for_xgboost.__name__))
            raise Exception(xg_boost_clf_exception.error_message_detail(str(e), sys)) from e

    def get_best_model(self, train_x, train_y, test_x, test_y, cluster_no=None):
        """
                                                Method Name: get_best_model
                                                Description: Find out the Model which has the best AUC score.
                                                Output: The best model name and the model object
                                                On Failure: Raise Exception

                                                Written By: iNeuron Intelligence
                                                Version: 1.0
                                                Revisions: None

                                        """

        # create best model for XGBoost
        try:
            if cluster_no is not None:
                title_generator = " Cluster " + cluster_no + " model {}"
            else:
                title_generator = "Model {}"

            # XG Boost model

            self.model_name.append('XG_BOOST')
            title = title_generator.format('XG_BOOST')
            self.logger_object.log('Entered the get_best_model method of the Model_Finder class')
            xgboost = self.get_best_params_for_xgboost(train_x, train_y)
            prediction_xgboost = xgboost.predict(test_x)  # Predictions using the XGBoost Model

            if len(test_y.unique()) == 1:  # if there is only one label in y, then roc_auc_score returns error. We
                # will use accuracy in that case
                xgboost_score = accuracy_score(test_y, prediction_xgboost)
                self.logger_object.log('Accuracy for XGBoost:' + str(xgboost_score))  # Log AUC
            else:
                xgboost_score = roc_auc_score(test_y, prediction_xgboost)  # AUC for XGBoost
                self.logger_object.log('AUC for XGBoost:' + str(xgboost_score))  # Log AUC
                y_score = xgboost.predict_proba(test_x)[:, 1]
                fpr, tpr, thresholds = roc_curve(test_y, y_score, pos_label=test_y.unique()[1])
                AccurayGraph().save_roc_curve_plot_binary_classification(fpr, tpr, self.project_id,
                                                                         execution_id=self.logger_object.execution_id,
                                                                         file_object=self.file_object,
                                                                         title=title)

            self.model.append(xgboost)
            self.score.append(xgboost_score)

            # create best model for naive bayes
            self.model_name.append('NAIVE_BAYES')
            title = title_generator.format('NAIVE_BAYES')
            naive_bayes = self.get_best_params_for_naive_bayes(train_x, train_y)
            prediction_naive_bayes = naive_bayes.predict(test_x)  # prediction using the Random Forest Algorithm
            self.model.append(naive_bayes)
            if len(test_y.unique()) == 1:  # if there is only one label in y,
                # then roc_auc_score returns error. We will use accuracy in that case
                naive_bayes_score = accuracy_score(test_y, prediction_naive_bayes)
                self.logger_object.log('Accuracy for naive bayes score' + str(naive_bayes_score))
            else:
                naive_bayes_score = roc_auc_score(test_y, prediction_naive_bayes)  # AUC for Random Forest
                self.logger_object.log('AUC for naive bayes score:' + str(naive_bayes_score))
                y_score = naive_bayes.predict_proba(test_x)[:, 1]
                fpr, tpr, thresholds = roc_curve(test_y, y_score, pos_label=test_y.unique()[0])
                AccurayGraph().save_roc_curve_plot_binary_classification(fpr, tpr, self.project_id,
                                                                         execution_id=self.logger_object.execution_id,
                                                                         file_object=self.file_object,
                                                                         title=title)

            self.score.append(naive_bayes_score)
            # create best model for Random forest
            self.model_name.append('Random_Forest')
            title = title_generator.format('Random_Forest')
            random_forest = self.get_best_params_for_random_forest(train_x, train_y)
            prediction_random_forest = random_forest.predict(test_x)
            self.model.append(random_forest)
            if len(test_y.unique()) == 1:
                random_forest_score = accuracy_score(test_y, prediction_random_forest)
                self.logger_object.log('Accuracy for Random Forest' + str(random_forest_score))
            else:
                random_forest_score = roc_auc_score(test_y, prediction_random_forest)  # AUC for Random Forest
                self.logger_object.log('AUC for Random Forest' + str(random_forest_score))
                y_score = random_forest.predict_proba(test_x)[:, 1]
                fpr, tpr, thresholds = roc_curve(test_y, y_score, pos_label=test_y.unique()[1])
                AccurayGraph().save_roc_curve_plot_binary_classification(fpr, tpr, self.project_id,
                                                                         execution_id=self.logger_object.execution_id,
                                                                         file_object=self.file_object,
                                                                         title=title)

            self.score.append(random_forest_score)

            # create best model for KNN
            self.model_name.append('KNN')
            title = title_generator.format('KNN')
            knn_clf = self.get_best_params_for_KNN(train_x, train_y)
            prediction_knn = knn_clf.predict(test_x)
            self.model.append(knn_clf)
            if len(test_y.unique()) == 1:
                knn_score = accuracy_score(test_y, prediction_knn)
                self.logger_object.log('Accuracy for KNN clf' + str(knn_score))
            else:
                knn_score = roc_auc_score(test_y, prediction_knn)  # AUC for Random Forest
                self.logger_object.log('AUC for KNN' + str(knn_score))
                y_score = knn_clf.predict_proba(test_x)[:, 1]
                fpr, tpr, thresholds = roc_curve(test_y, y_score, pos_label=test_y.unique()[1])
                AccurayGraph().save_roc_curve_plot_binary_classification(fpr, tpr, self.project_id,
                                                                         execution_id=self.logger_object.execution_id,
                                                                         file_object=self.file_object,
                                                                         title=title)
            self.score.append(knn_score)

            """ 5. SVC """
            if len(test_y.unique()) != 1:
                self.model_name.append("SVC")
                title = title_generator.format("SVC")
                svc_clf = self.get_best_params_for_svm_fraud_detection_and_scania(train_x, train_y)
                prediction_svc = svc_clf.predict(test_x)
                self.model.append(svc_clf)
                if len(test_y.unique()) == 1:
                    svc_score = accuracy_score(test_y, prediction_svc)
                    self.logger_object.log('Accuracy for svc clf' + str(svc_score))
                else:
                    svc_score = roc_auc_score(test_y, prediction_svc)  # AUC for Random Forest
                    self.logger_object.log('AUC for svc' + str(svc_score))
                    y_score = svc_clf.predict_proba(test_x)[:, 1]
                    fpr, tpr, thresholds = roc_curve(test_y, y_score, pos_label=test_y.unique()[1])
                    AccurayGraph().save_roc_curve_plot_binary_classification(fpr, tpr, self.project_id,
                                                                             execution_id=self.logger_object.execution_id,
                                                                             file_object=self.file_object,
                                                                             title=title)
                self.score.append(svc_score)

            AccurayGraph().save_accuracy_bar_graph(
                model_name_list=self.model_name,
                accuracy_score_list=self.score,
                project_id=self.project_id,
                execution_id=self.logger_object.execution_id,
                file_object=self.file_object,
                x_label="Model List",
                y_label="Accuracy score comparison {}".format(self.model_name),
                title="Accuracy Score "
            )
            execution_model_comparison_id = str(uuid.uuid4())
            for data in zip(self.model_name, self.score):
                self.save_accuracy_data(model_name=data[0], score=data[1],
                                        execution_model_comparision_id=execution_model_comparison_id)
            # comparing the two models
            return self.get_best_model_on_score(model_name=self.model_name, model=self.model, score=self.score)

        except Exception as e:
            model_finder = ModelFinderException(
                "Model Selection Failed in module [{0}] class [{1}] method [{2}]"
                    .format(self.__module__, ModelFinder.__name__,
                            self.get_best_model.__name__))
            raise Exception(model_finder.error_message_detail(str(e), sys)) from e

    def get_best_params_for_random_forest_thyroid(self, train_x, train_y):
        """
                                Method Name: get_best_params_for_random_forest
                                Description: get the parameters for Random Forest Algorithm which give the best accuracy.
                                             Use Hyper Parameter Tuning.
                                Output: The model with the best parameters
                                On Failure: Raise Exception

                                Written By: iNeuron Intelligence
                                Version: 1.0
                                Revisions: None

                        """
        self.logger_object.log('Entered the get_best_params_for_random_forest method of the Model_Finder class')
        try:
            # initializing with different combination of parameters
            param_grid = {"n_estimators": [10, 50, 100, 130], "criterion": ['gini', 'entropy'],
                          "max_depth": range(2, 4, 1), "max_features": ['auto', 'log2']}

            # Creating an object of the Grid Search class
            grid = GridSearchCV(estimator=RandomForestClassifier(), param_grid=param_grid, cv=5, verbose=3)
            # finding the best parameters
            grid.fit(train_x, train_y)

            # extracting the best parameters
            criterion = grid.best_params_['criterion']
            max_depth = grid.best_params_['max_depth']
            max_features = grid.best_params_['max_features']
            n_estimators = grid.best_params_['n_estimators']

            # creating a new model with the best parameters
            clf = RandomForestClassifier(n_estimators=n_estimators, criterion=criterion,
                                         max_depth=max_depth, max_features=max_features)
            # training the mew model
            clf.fit(train_x, train_y)
            self.logger_object.log('Random Forest best params: ' + str(
                grid.best_params_) + '. Exited the get_best_params_for_random_forest method of the Model_Finder class')

            return clf
        except Exception as e:
            model_finder = ModelFinderException(
                "Model Selection Failed in module [{0}] class [{1}] method [{2}]"
                    .format(self.__module__, ModelFinder.__name__,
                            self.get_best_params_for_random_forest_thyroid.__name__))
            raise Exception(model_finder.error_message_detail(str(e), sys)) from e

    def get_best_params_for_KNN_fraud_detection(self, train_x, train_y):
        """
                                                Method Name: get_best_params_for_KNN
                                                Description: get the parameters for KNN Algorithm which give the best accuracy.
                                                             Use Hyper Parameter Tuning.
                                                Output: The model with the best parameters
                                                On Failure: Raise Exception

                                                Written By: iNeuron Intelligence
                                                Version: 1.0
                                                Revisions: None

                                        """
        self.logger_object.log('Entered the get_best_params_for_Ensembled_KNN method of the Model_Finder class')
        try:
            # initializing with different combination of parameters
            param_grid_knn = {
                'algorithm': ['ball_tree', 'kd_tree', 'brute'],
                'leaf_size': [10, 17, 24, 28, 30, 35],
                'n_neighbors': [4, 5],
                'p': [1, 2]
            }

            # Creating an object of the Grid Search class
            grid = GridSearchCV(KNeighborsClassifier(), param_grid_knn, verbose=3,
                                cv=5)
            # finding the best parameters
            grid.fit(train_x, train_y)

            # extracting the best parameters
            algorithm = grid.best_params_['algorithm']
            leaf_size = grid.best_params_['leaf_size']
            n_neighbors = grid.best_params_['n_neighbors']
            p = grid.best_params_['p']

            # creating a new model with the best parameters
            knn = KNeighborsClassifier(algorithm=algorithm, leaf_size=leaf_size,
                                       n_neighbors=n_neighbors, p=p, n_jobs=-1)
            # training the mew model
            knn.fit(train_x, train_y)
            self.logger_object.log('KNN best params: ' + str(
                grid.best_params_) + '. Exited the KNN method of the Model_Finder class')
            return knn
        except Exception as e:
            model_finder = ModelFinderException(
                "Model Selection Failed in module [{0}] class [{1}] method [{2}]"
                    .format(self.__module__, ModelFinder.__name__,
                            self.get_best_params_for_KNN_fraud_detection.__name__))
            raise Exception(model_finder.error_message_detail(str(e), sys)) from e

    def get_best_params_for_KNN(self, train_x, train_y):
        """
                                                Method Name: get_best_params_for_KNN
                                                Description: get the parameters for KNN Algorithm which give the best accuracy.
                                                             Use Hyper Parameter Tuning.
                                                Output: The model with the best parameters
                                                On Failure: Raise Exception

                                                Written By: iNeuron Intelligence
                                                Version: 1.0
                                                Revisions: None

                                        """
        self.logger_object.log('Entered the get_best_params_for_Ensembled_KNN method of the Model_Finder class')
        try:
            # initializing with different combination of parameters
            param_grid_knn = {
                'algorithm': ['ball_tree', 'kd_tree', 'brute'],
                'leaf_size': [10, 17, 24, 28, 30, 35],
                'n_neighbors': [4, 5, 8, 10, 11],
                'p': [1, 2]
            }

            # Creating an object of the Grid Search class
            grid = GridSearchCV(KNeighborsClassifier(), param_grid_knn, verbose=3,
                                cv=5)
            # finding the best parameters
            grid.fit(train_x, train_y)

            # extracting the best parameters
            algorithm = grid.best_params_['algorithm']
            leaf_size = grid.best_params_['leaf_size']
            n_neighbors = grid.best_params_['n_neighbors']
            p = grid.best_params_['p']

            # creating a new model with the best parameters
            knn = KNeighborsClassifier(algorithm=algorithm, leaf_size=leaf_size,
                                       n_neighbors=n_neighbors, p=p, n_jobs=-1)
            # training the mew model
            knn.fit(train_x, train_y)
            self.logger_object.log('KNN best params: ' + str(
                grid.best_params_) + '. Exited the KNN method of the Model_Finder class')
            return knn
        except Exception as e:
            model_finder = ModelFinderException(
                "Model Selection Failed in module [{0}] class [{1}] method [{2}]"
                    .format(self.__module__, ModelFinder.__name__,
                            self.get_best_params_for_KNN.__name__))
            raise Exception(model_finder.error_message_detail(str(e), sys)) from e

    def get_best_model_thyroid(self, train_x, train_y, test_x, test_y, cluster_no=None):
        """
                                                Method Name: get_best_model
                                                Description: Find out the Model which has the best AUC score.
                                                Output: The best model name and the model object
                                                On Failure: Raise Exception

                                                Written By: iNeuron Intelligence
                                                Version: 1.0
                                                Revisions: None

                                        """

        # create best model for KNN
        try:
            self.logger_object.log('Entered the get_best_model method of the Model_Finder class')
            if cluster_no is not None:
                title_generator = " Cluster " + cluster_no + " model {}"
            else:
                title_generator = "Model {}"
            # XG Boost model
            self.model_name.append('XG_BOOST')
            title = title_generator.format('XG_BOOST')
            self.logger_object.log('Entered the get_best_model method of the Model_Finder class')
            xgboost = self.get_best_params_for_xgboost(train_x, train_y)
            prediction_xgboost = xgboost.predict(test_x)  # Predictions using the XGBoost Model

            if len(test_y.unique()) == 1:  # if there is only one label in y, then roc_auc_score returns error. We
                # will use accuracy in that case
                xgboost_score = accuracy_score(test_y, prediction_xgboost)
                self.logger_object.log('Accuracy for XGBoost:' + str(xgboost_score))  # Log AUC
            else:
                y_scores = xgboost.predict_proba(test_x)
                AccurayGraph().save_plot_multiclass_roc_curve(test_y, y_scores, xgboost,
                                                              project_id=self.project_id,
                                                              execution_id=self.logger_object.execution_id,
                                                              file_object=self.file_object,
                                                              title=title
                                                              )
                xgboost_score = roc_auc_score(test_y, y_scores, multi_class='ovr')  # AUC for XGBoost
                self.logger_object.log('AUC for XGBoost:' + str(xgboost_score))  # Log AUC

            self.model.append(xgboost)
            self.score.append(xgboost_score)

            # create best model for naive bayes
            self.model_name.append('NAIVE_BAYES')
            title = title_generator.format('NAIVE_BAYES')
            naive_bayes = self.get_best_params_for_naive_bayes(train_x, train_y)
            prediction_naive_bayes = naive_bayes.predict(test_x)  # prediction using the Random Forest Algorithm
            self.model.append(naive_bayes)
            if len(test_y.unique()) == 1:  # if there is only one label in y,
                # then roc_auc_score returns error. We will use accuracy in that case
                naive_bayes_score = accuracy_score(test_y, prediction_naive_bayes)
                self.logger_object.log('Accuracy for naive bayes score' + str(naive_bayes_score))
            else:
                y_scores = naive_bayes.predict_proba(test_x)
                AccurayGraph().save_plot_multiclass_roc_curve(test_y, y_scores, naive_bayes,
                                                              project_id=self.project_id,
                                                              execution_id=self.logger_object.execution_id,
                                                              file_object=self.file_object,
                                                              title=title
                                                              )
                naive_bayes_score = roc_auc_score(test_y, y_scores,
                                                  multi_class='ovr')  # AUC for Random Forest
                self.logger_object.log('AUC for naive bayes score:' + str(naive_bayes_score))

            self.score.append(naive_bayes_score)
            # create best model for Random forest
            self.model_name.append('Random_Forest')
            title = title_generator.format('Random_Forest')
            random_forest = self.get_best_params_for_random_forest_thyroid(train_x, train_y)
            prediction_random_forest = random_forest.predict(test_x)
            self.model.append(random_forest)
            if len(test_y.unique()) == 1:
                random_forest_score = accuracy_score(test_y, prediction_random_forest)
                self.logger_object.log('Accuracy for Random Forest' + str(random_forest_score))
            else:
                y_scores = random_forest.predict_proba(test_x)
                AccurayGraph().save_plot_multiclass_roc_curve(test_y, y_scores, random_forest,
                                                              project_id=self.project_id,
                                                              execution_id=self.logger_object.execution_id,
                                                              file_object=self.file_object,
                                                              title=title
                                                              )
                random_forest_score = roc_auc_score(test_y, y_scores,
                                                    multi_class='ovr')  # AUC for Random Forest
                self.logger_object.log('AUC for Random Forest' + str(random_forest_score))

            self.score.append(random_forest_score)

            # create best model for KNN
            self.model_name.append('KNN')
            title = title_generator.format('KNN')
            knn_clf = self.get_best_params_for_KNN(train_x, train_y)
            prediction_knn = knn_clf.predict(test_x)
            self.model.append(knn_clf)
            if len(test_y.unique()) == 1:
                knn_score = accuracy_score(test_y, prediction_knn)
                self.logger_object.log('Accuracy for KNN clf' + str(knn_score))
            else:
                y_scores = knn_clf.predict_proba(test_x)
                AccurayGraph().save_plot_multiclass_roc_curve(test_y, y_scores, knn_clf,
                                                              project_id=self.project_id,
                                                              execution_id=self.logger_object.execution_id,
                                                              file_object=self.file_object,
                                                              title=title
                                                              )
                knn_score = roc_auc_score(test_y, y_scores, multi_class='ovr')  # AUC for Random Forest
                self.logger_object.log('AUC for KNN' + str(knn_score))

            self.score.append(knn_score)

            """ 5. SVC """
            if len(test_y.unique()) != 1:
                self.model_name.append("SVC")
                title = title_generator.format("SVC")
                svc_clf = self.get_best_params_for_svm_fraud_detection_and_scania(train_x, train_y)
                prediction_svc = svc_clf.predict(test_x)
                self.model.append(svc_clf)
                if len(test_y.unique()) == 1:
                    svc_score = accuracy_score(test_y, prediction_svc)
                    self.logger_object.log('Accuracy for svc clf' + str(svc_score))
                else:
                    y_scores = svc_clf.predict_proba(test_x)
                    AccurayGraph().save_plot_multiclass_roc_curve(test_y, y_scores, svc_clf,
                                                                  project_id=self.project_id,
                                                                  execution_id=self.logger_object.execution_id,
                                                                  file_object=self.file_object,
                                                                  title=title
                                                                  )
                    svc_score = roc_auc_score(test_y, y_scores, multi_class='ovr')  # AUC for Random Forest
                    self.logger_object.log('AUC for svc' + str(svc_score))

                self.score.append(svc_score)

            AccurayGraph().save_accuracy_bar_graph(
                model_name_list=self.model_name,
                accuracy_score_list=self.score,
                project_id=self.project_id,
                execution_id=self.logger_object.execution_id,
                file_object=self.file_object,
                x_label="Model List",
                y_label="Accuracy score comparison {}".format(self.model_name),
                title="Accuracy Score "
            )

            execution_model_comparison_id = str(uuid.uuid4())

            for data in zip(self.model_name, self.score):
                self.save_accuracy_data(model_name=data[0], score=data[1],
                                        execution_model_comparision_id=execution_model_comparison_id)
            return self.get_best_model_on_score(model_name=self.model_name, model=self.model, score=self.score)

        except Exception as e:
            model_finder = ModelFinderException(
                "Model Selection Failed in module [{0}] class [{1}] method [{2}]"
                    .format(self.__module__, ModelFinder.__name__,
                            self.get_best_model_thyroid.__name__))
            raise Exception(model_finder.error_message_detail(str(e), sys)) from e

    def get_best_params_for_random_forest_mushroom(self, train_x, train_y):
        """
                                Method Name: get_best_params_for_random_forest
                                Description: get the parameters for Random Forest Algorithm which give the best accuracy.
                                             Use Hyper Parameter Tuning.
                                Output: The model with the best parameters
                                On Failure: Raise Exception

                                Written By: iNeuron Intelligence
                                Version: 1.0
                                Revisions: None

                        """

        try:
            self.logger_object.log('Entered the get_best_params_for_random_forest method of the Model_Finder class')
            # initializing with different combination of parameters
            param_grid = {"n_estimators": [10, 50, 100, 130], "criterion": ['gini', 'entropy'],
                          "max_depth": range(2, 4, 1), "max_features": ['auto', 'log2']}

            # Creating an object of the Grid Search class
            grid = GridSearchCV(estimator=RandomForestClassifier(), param_grid=param_grid, cv=5, verbose=3)
            # finding the best parameters
            grid.fit(train_x, train_y)

            # extracting the best parameters
            criterion = grid.best_params_['criterion']
            max_depth = grid.best_params_['max_depth']
            max_features = grid.best_params_['max_features']
            n_estimators = grid.best_params_['n_estimators']

            # creating a new model with the best parameters
            clf = RandomForestClassifier(n_estimators=n_estimators, criterion=criterion,
                                         max_depth=max_depth, max_features=max_features)
            # training the mew model
            clf.fit(train_x, train_y)
            self.logger_object.log('Random Forest best params: ' + str(
                grid.best_params_) + '.Exited the get_best_params_for_random_forest method of the Model_Finder class')

            return clf
        except Exception as e:
            model_finder = ModelFinderException(
                "Model Selection Failed in module [{0}] class [{1}] method [{2}]"
                    .format(self.__module__, ModelFinder.__name__,
                            self.get_best_params_for_random_forest_mushroom.__name__))
            raise Exception(model_finder.error_message_detail(str(e), sys)) from e

    def get_best_params_for_KNN_mushroom(self, train_x, train_y):
        """
                                                Method Name: get_best_params_for_KNN
                                                Description: get the parameters for KNN Algorithm which give the best accuracy.
                                                             Use Hyper Parameter Tuning.
                                                Output: The model with the best parameters
                                                On Failure: Raise Exception

                                                Written By: iNeuron Intelligence
                                                Version: 1.0
                                                Revisions: None

                                        """

        try:
            self.logger_object.log('Entered the get_best_params_for_KNN method of the Model_Finder class')
            # initializing with different combination of parameters
            param_grid_knn = {
                'algorithm': ['ball_tree', 'kd_tree', 'brute'],
                'leaf_size': [10, 17, 24, 28, 30, 35],
                'n_neighbors': [4, 5, 8, 10, 11],
                'p': [1, 2]
            }

            # Creating an object of the Grid Search class
            grid = GridSearchCV(KNeighborsClassifier(), param_grid_knn, verbose=3,
                                cv=5)
            # finding the best parameters
            grid.fit(train_x, train_y)

            # extracting the best parameters
            algorithm = grid.best_params_['algorithm']
            leaf_size = grid.best_params_['leaf_size']
            n_neighbors = grid.best_params_['n_neighbors']
            p = grid.best_params_['p']

            # creating a new model with the best parameters
            knn = KNeighborsClassifier(algorithm=algorithm, leaf_size=leaf_size,
                                       n_neighbors=n_neighbors, p=p, n_jobs=-1)
            # training the mew model
            knn.fit(train_x, train_y)
            self.logger_object.log('KNN best params: ' + str(
                grid.best_params_) + '. Exited the KNN method of the Model_Finder class')
            return knn
        except Exception as e:
            model_finder = ModelFinderException(
                "Model Selection Failed in module [{0}] class [{1}] method [{2}]"
                    .format(self.__module__, ModelFinder.__name__,
                            self.get_best_params_for_KNN_mushroom.__name__))
            raise Exception(model_finder.error_message_detail(str(e), sys)) from e

    def get_binary_format_target_value(self, target_column):
        try:
            column_value = target_column.unique()
            target_column = target_column.replace(column_value[0], 0)
            target_column = target_column.replace(column_value[1], 1)
            return target_column
        except Exception as e:
            model_finder = ModelFinderException(
                "Model Selection Failed in module [{0}] class [{1}] method [{2}]"
                    .format(self.__module__, ModelFinder.__name__,
                            self.get_best_params_for_KNN_mushroom.__name__))
            raise Exception(model_finder.error_message_detail(str(e), sys)) from e

    def get_best_model_mushroom(self, train_x, train_y, test_x, test_y, cluster_no=None):
        """
                                                Method Name: get_best_model
                                                Description: Find out the Model which has the best AUC score.
                                                Output: The best model name and the model object
                                                On Failure: Raise Exception

                                                Written By: iNeuron Intelligence
                                                Version: 1.0
                                                Revisions: None

                                        """

        # create best model for KNN
        try:
            self.logger_object.log('Entered the get_best_model method of the Model_Finder class')
            title_generator = " Cluster " + cluster_no + " model {}"

            # XG Boost model

            self.model_name.append('XG_BOOST')
            title = title_generator.format('XG_BOOST')

            xgboost = self.get_best_params_for_xgboost_income_prediction(train_x, train_y)
            prediction_xgboost = xgboost.predict(test_x)  # Predictions using the XGBoost Model

            if len(test_y.unique()) == 1:  # if there is only one label in y, then roc_auc_score returns error. We
                # will use accuracy in that case
                xgboost_score = accuracy_score(test_y, prediction_xgboost)
                self.logger_object.log('Accuracy for XGBoost:' + str(xgboost_score))  # Log AUC
            else:
                xgboost_score = roc_auc_score(test_y, prediction_xgboost)  # AUC for XGBoost
                self.logger_object.log('AUC for XGBoost:' + str(xgboost_score))  # Log AUC
                y_score = xgboost.predict_proba(test_x)[:, 1]
                fpr, tpr, thresholds = roc_curve(test_y, y_score, pos_label=test_y.unique()[1])
                AccurayGraph().save_roc_curve_plot_binary_classification(fpr, tpr, self.project_id,
                                                                         execution_id=self.logger_object.execution_id,
                                                                         file_object=self.file_object,
                                                                         title=title)

            self.model.append(xgboost)
            self.score.append(xgboost_score)

            # create best model for naive bayes
            self.model_name.append('NAIVE_BAYES')
            title = title_generator.format('NAIVE_BAYES')
            naive_bayes = self.get_best_params_for_naive_bayes(train_x, train_y)
            prediction_naive_bayes = naive_bayes.predict(test_x)  # prediction using the Random Forest Algorithm
            self.model.append(naive_bayes)
            if len(test_y.unique()) == 1:  # if there is only one label in y,
                # then roc_auc_score returns error. We will use accuracy in that case
                naive_bayes_score = accuracy_score(test_y, prediction_naive_bayes)
                self.logger_object.log('Accuracy for naive bayes score' + str(naive_bayes_score))
            else:
                naive_bayes_score = roc_auc_score(test_y, prediction_naive_bayes)  # AUC for Random Forest
                self.logger_object.log('AUC for naive bayes score:' + str(naive_bayes_score))
                y_score = naive_bayes.predict_proba(test_x)[:, 1]
                fpr, tpr, thresholds = roc_curve(test_y, y_score, pos_label=test_y.unique()[0])
                AccurayGraph().save_roc_curve_plot_binary_classification(fpr, tpr, self.project_id,
                                                                         execution_id=self.logger_object.execution_id,
                                                                         file_object=self.file_object,
                                                                         title=title)

            self.score.append(naive_bayes_score)
            # create best model for Random forest
            self.model_name.append('Random_Forest')
            title = title_generator.format('Random_Forest')
            random_forest = self.get_best_params_for_random_forest_mushroom(train_x, train_y)
            prediction_random_forest = random_forest.predict(test_x)
            self.model.append(random_forest)
            if len(test_y.unique()) == 1:
                random_forest_score = accuracy_score(test_y, prediction_random_forest)
                self.logger_object.log('Accuracy for Random Forest' + str(random_forest_score))
            else:
                random_forest_score = roc_auc_score(test_y, prediction_random_forest)  # AUC for Random Forest
                self.logger_object.log('AUC for Random Forest' + str(random_forest_score))
                y_score = random_forest.predict_proba(test_x)[:, 1]
                fpr, tpr, thresholds = roc_curve(test_y, y_score, pos_label=test_y.unique()[1])
                AccurayGraph().save_roc_curve_plot_binary_classification(fpr, tpr, self.project_id,
                                                                         execution_id=self.logger_object.execution_id,
                                                                         file_object=self.file_object,
                                                                         title=title)

            self.score.append(random_forest_score)

            # create best model for KNN
            self.model_name.append('KNN')
            title = title_generator.format('KNN')
            knn_clf = self.get_best_params_for_KNN_mushroom(train_x, train_y)
            prediction_knn = knn_clf.predict(test_x)
            self.model.append(knn_clf)
            if len(test_y.unique()) == 1:
                knn_score = accuracy_score(test_y, prediction_knn)
                self.logger_object.log('Accuracy for KNN clf' + str(knn_score))
            else:
                knn_score = roc_auc_score(test_y, prediction_knn)  # AUC for Random Forest
                self.logger_object.log('AUC for KNN' + str(knn_score))
                y_score = knn_clf.predict_proba(test_x)[:, 1]
                fpr, tpr, thresholds = roc_curve(test_y, y_score, pos_label=test_y.unique()[1])
                AccurayGraph().save_roc_curve_plot_binary_classification(fpr, tpr, self.project_id,
                                                                         execution_id=self.logger_object.execution_id,
                                                                         file_object=self.file_object,
                                                                         title=title)
            self.score.append(knn_score)

            if len(test_y.unique()) != 1:
                """ 5. SVC """
                self.model_name.append("SVC")
                title = title_generator.format("SVC")
                svc_clf = self.get_best_params_for_svm_fraud_detection_and_scania(train_x, train_y)
                prediction_svc = svc_clf.predict(test_x)
                self.model.append(svc_clf)
                if len(test_y.unique()) == 1:
                    svc_score = accuracy_score(test_y, prediction_svc)
                    self.logger_object.log('Accuracy for svc clf' + str(svc_score))
                else:
                    svc_score = roc_auc_score(test_y, prediction_svc)  # AUC for Random Forest
                    self.logger_object.log('AUC for svc' + str(svc_score))
                    y_score = svc_clf.predict_proba(test_x)[:, 1]
                    fpr, tpr, thresholds = roc_curve(test_y, y_score, pos_label=test_y.unique()[1])
                    AccurayGraph().save_roc_curve_plot_binary_classification(fpr, tpr, self.project_id,
                                                                             execution_id=self.logger_object.execution_id,
                                                                             file_object=self.file_object,
                                                                             title=title)
                self.score.append(svc_score)

            AccurayGraph().save_accuracy_bar_graph(
                model_name_list=self.model_name,
                accuracy_score_list=self.score,
                project_id=self.project_id,
                execution_id=self.logger_object.execution_id,
                file_object=self.file_object,
                x_label="Model List",
                y_label="Accuracy score comparison {}".format(self.model_name),
                title="Cluster " + str(cluster_no) + "Accuracy Score "
            )
            execution_model_comparison_id = str(uuid.uuid4())

            for data in zip(self.model_name, self.score):
                self.save_accuracy_data(model_name=data[0], score=data[1],
                                        execution_model_comparision_id=execution_model_comparison_id)
            return self.get_best_model_on_score(model_name=self.model_name, model=self.model, score=self.score)
        except Exception as e:
            model_finder = ModelFinderException(
                "Model Selection Failed in module [{0}] class [{1}] method [{2}]"
                    .format(self.__module__, ModelFinder.__name__,
                            self.get_best_model_mushroom.__name__))
            raise Exception(model_finder.error_message_detail(str(e), sys)) from e

    def save_accuracy_data(self, model_name, score, execution_model_comparision_id):
        try:
            accuracy_graph_data = AccurayGraph(project_id=self.project_id,
                                               model_accuracy_dict={'model_name': model_name,
                                                                    'score': score,
                                                                    'execution_model_comparision': execution_model_comparision_id,
                                                                    'training_execution_id': self.logger_object.execution_id}
                                               )
            accuracy_graph_data.save_accuracy()
        except Exception as e:
            model_finder = ModelFinderException(
                "save model accuracy [{0}] class [{1}] method [{2}]"
                    .format(self.__module__, ModelFinder.__name__,
                            self.get_best_model_mushroom.__name__))
            raise Exception(model_finder.error_message_detail(str(e), sys)) from e

    def get_best_params_for_svm_fraud_detection_and_scania(self, train_x, train_y):
        """
        Method Name: get_best_params_for_naive_bayes
        Description: get the parameters for the SVM Algorithm which give the best accuracy.
                     Use Hyper Parameter Tuning.
        Output: The model with the best parameters
        On Failure: Raise Exception

        Written By: iNeuron Intelligence
        Version: 1.0
        Revisions: None

        """
        try:
            self.logger_object.log('Entered the get_best_params_for_svm method of the Model_Finder class')

            # initializing with different combination of parameters
            param_grid = {"kernel": ['rbf', 'sigmoid'],
                          "C": [0.1, 0.5, 1.0],
                          "random_state": [0, 100, 200, 300]}

            # Creating an object of the Grid Search class
            grid = GridSearchCV(estimator=SVC(), param_grid=param_grid, cv=5, verbose=3)
            # finding the best parameters
            grid.fit(train_x, train_y)

            # extracting the best parameters
            kernel = grid.best_params_['kernel']
            C = grid.best_params_['C']
            random_state = grid.best_params_['random_state']

            # creating a new model with the best parameters
            sv_classifier = SVC(kernel=kernel, C=C, random_state=random_state, probability=True)
            # training the mew model
            sv_classifier.fit(train_x, train_y)
            self.logger_object.log('SVM best params: ' + str(
                grid.best_params_) + '. Exited the get_best_params_for_svm method of the Model_Finder class')
            return sv_classifier
        except Exception as e:
            model_finder = ModelFinderException(
                "Failed in [{0}] class [{1}] method [{2}]"
                    .format(self.__module__, ModelFinder.__name__,
                            self.get_best_params_for_svm_fraud_detection_and_scania.__name__))
            raise Exception(model_finder.error_message_detail(str(e), sys)) from e

    def get_best_params_for_xgboost_fraud_detection(self, train_x, train_y):
        """
        Method Name: get_best_params_for_xgboost
        Description: get the parameters for XGBoost Algorithm which give the best accuracy.
        Use Hyper Parameter Tuning.
        Output: The model with the best parameters
        On Failure: Raise Exception

        Written By: iNeuron Intelligence
        Version: 1.0
        Revisions: None

        """
        try:
            # initializing with different combination of parameters
            self.logger_object.log('Entered the get_best_params_for_xgboost method of the Model_Finder class')
            param_grid_xgboost = {
                "n_estimators": [100, 130], "criterion": ['gini', 'entropy'],

                "max_depth": range(8, 10, 1)

            }
            # Creating an object of the Grid Search class
            grid = GridSearchCV(XGBClassifier(objective='binary:logistic'), param_grid_xgboost, verbose=3,
                                cv=5)
            # finding the best parameters
            grid.fit(train_x, train_y)

            # extracting the best parameters
            criterion = grid.best_params_['criterion']
            max_depth = grid.best_params_['max_depth']
            n_estimators = grid.best_params_['n_estimators']

            # creating a new model with the best parameters
            xgb = XGBClassifier(criterion=criterion, max_depth=max_depth, n_estimators=n_estimators,
                                n_jobs=-1)
            # training the mew model
            xgb.fit(train_x, train_y)
            self.logger_object.log('XGBoost best params: ' + str(
                grid.best_params_) + '. Exited the get_best_params_for_xgboost method of the Model_Finder class')
            return xgb
        except Exception as e:
            model_finder = ModelFinderException(
                "Failed in [{0}] class [{1}] method [{2}]"
                    .format(self.__module__, ModelFinder.__name__,
                            self.get_best_params_for_xgboost_fraud_detection.__name__))
            raise Exception(model_finder.error_message_detail(str(e), sys)) from e

    def get_best_model_fraud_detection(self, train_x, train_y, test_x, test_y, cluster_no=None):
        """
        Method Name: get_best_model
        Description: Find out the Model which has the best AUC score.
        Output: The best model name and the model object
        On Failure: Raise Exception

        Written By: iNeuron Intelligence
        Version: 1.0
        Revisions: None

        """

        # create best model for XGBoost
        try:
            self.logger_object.log('Entered the get_best_model method of the Model_Finder class')

            title_generator = " Cluster " + cluster_no + " model {}"

            # XG Boost model

            self.model_name.append('XG_BOOST')
            title = title_generator.format('XG_BOOST')
            self.logger_object.log('Entered the get_best_model method of the Model_Finder class')
            xgboost = self.get_best_params_for_xgboost_fraud_detection(train_x, train_y)
            prediction_xgboost = xgboost.predict(test_x)  # Predictions using the XGBoost Model

            if len(test_y.unique()) == 1:  # if there is only one label in y, then roc_auc_score returns error. We
                # will use accuracy in that case
                xgboost_score = accuracy_score(test_y, prediction_xgboost)
                self.logger_object.log('Accuracy for XGBoost:' + str(xgboost_score))  # Log AUC
            else:
                xgboost_score = roc_auc_score(test_y, prediction_xgboost)  # AUC for XGBoost
                self.logger_object.log('AUC for XGBoost:' + str(xgboost_score))  # Log AUC
                y_score = xgboost.predict_proba(test_x)[:, 1]
                fpr, tpr, thresholds = roc_curve(test_y, y_score, pos_label=test_y.unique()[1])
                AccurayGraph().save_roc_curve_plot_binary_classification(fpr, tpr, self.project_id,
                                                                         execution_id=self.logger_object.execution_id,
                                                                         file_object=self.file_object,
                                                                         title=title)

            self.model.append(xgboost)
            self.score.append(xgboost_score)

            # create best model for naive bayes
            self.model_name.append('NAIVE_BAYES')
            title = title_generator.format('NAIVE_BAYES')
            naive_bayes = self.get_best_params_for_naive_bayes(train_x, train_y)
            prediction_naive_bayes = naive_bayes.predict(test_x)  # prediction using the Random Forest Algorithm
            self.model.append(naive_bayes)
            if len(test_y.unique()) == 1:  # if there is only one label in y,
                # then roc_auc_score returns error. We will use accuracy in that case
                naive_bayes_score = accuracy_score(test_y, prediction_naive_bayes)
                self.logger_object.log('Accuracy for naive bayes score' + str(naive_bayes_score))
            else:
                naive_bayes_score = roc_auc_score(test_y, prediction_naive_bayes)  # AUC for Random Forest
                self.logger_object.log('AUC for naive bayes score:' + str(naive_bayes_score))
                y_score = naive_bayes.predict_proba(test_x)[:, 1]
                fpr, tpr, thresholds = roc_curve(test_y, y_score, pos_label=test_y.unique()[0])
                AccurayGraph().save_roc_curve_plot_binary_classification(fpr, tpr, self.project_id,
                                                                         execution_id=self.logger_object.execution_id,
                                                                         file_object=self.file_object,
                                                                         title=title)

            self.score.append(naive_bayes_score)
            # create best model for Random forest
            self.model_name.append('Random_Forest')
            title = title_generator.format('Random_Forest')
            random_forest = self.get_best_params_for_svm_fraud_detection_and_scania(train_x, train_y)
            prediction_random_forest = random_forest.predict(test_x)
            self.model.append(random_forest)
            if len(test_y.unique()) == 1:
                random_forest_score = accuracy_score(test_y, prediction_random_forest)
                self.logger_object.log('Accuracy for Random Forest' + str(random_forest_score))
            else:
                random_forest_score = roc_auc_score(test_y, prediction_random_forest)  # AUC for Random Forest
                self.logger_object.log('AUC for Random Forest' + str(random_forest_score))
                y_score = random_forest.predict_proba(test_x)[:, 1]
                fpr, tpr, thresholds = roc_curve(test_y, y_score, pos_label=test_y.unique()[1])
                AccurayGraph().save_roc_curve_plot_binary_classification(fpr, tpr, self.project_id,
                                                                         execution_id=self.logger_object.execution_id,
                                                                         file_object=self.file_object,
                                                                         title=title)

            self.score.append(random_forest_score)

            # create best model for KNN
            self.model_name.append('KNN')
            title = title_generator.format('KNN')
            knn_clf = self.get_best_params_for_KNN_fraud_detection(train_x, train_y)
            prediction_knn = knn_clf.predict(test_x)
            self.model.append(knn_clf)
            if len(test_y.unique()) == 1:
                knn_score = accuracy_score(test_y, prediction_knn)
                self.logger_object.log('Accuracy for KNN clf' + str(knn_score))
            else:
                knn_score = roc_auc_score(test_y, prediction_knn)  # AUC for Random Forest
                self.logger_object.log('AUC for KNN' + str(knn_score))
                y_score = knn_clf.predict_proba(test_x)[:, 1]
                fpr, tpr, thresholds = roc_curve(test_y, y_score, pos_label=test_y.unique()[1])
                AccurayGraph().save_roc_curve_plot_binary_classification(fpr, tpr, self.project_id,
                                                                         execution_id=self.logger_object.execution_id,
                                                                         file_object=self.file_object,
                                                                         title=title)
            self.score.append(knn_score)

            if len(test_y.unique()) != 1:
                """ 5. SVC """
                self.model_name.append("SVC")
                title = title_generator.format("SVC")
                svc_clf = self.get_best_params_for_svm_phising_classifier(train_x, train_y)
                prediction_svc = svc_clf.predict(test_x)
                self.model.append(svc_clf)
                if len(test_y.unique()) == 1:
                    svc_score = accuracy_score(test_y, prediction_svc)
                    self.logger_object.log('Accuracy for svc clf' + str(svc_score))
                else:
                    svc_score = roc_auc_score(test_y, prediction_svc)  # AUC for Random Forest
                    self.logger_object.log('AUC for svc' + str(svc_score))
                    y_score = svc_clf.predict_proba(test_x)[:, 1]
                    fpr, tpr, thresholds = roc_curve(test_y, y_score, pos_label=test_y.unique()[1])
                    AccurayGraph().save_roc_curve_plot_binary_classification(fpr, tpr, self.project_id,
                                                                             execution_id=self.logger_object.execution_id,
                                                                             file_object=self.file_object,
                                                                             title=title)
                self.score.append(svc_score)

            AccurayGraph().save_accuracy_bar_graph(
                model_name_list=self.model_name,
                accuracy_score_list=self.score,
                project_id=self.project_id,
                execution_id=self.logger_object.execution_id,
                file_object=self.file_object,
                x_label="Model List",
                y_label="Accuracy score comparison {}".format(self.model_name),
                title="Cluster " + str(cluster_no) + "Accuracy Score "
            )
            execution_model_comparison_id = str(uuid.uuid4())

            for data in zip(self.model_name, self.score):
                self.save_accuracy_data(model_name=data[0], score=data[1],
                                        execution_model_comparision_id=execution_model_comparison_id)

            return self.get_best_model_on_score(model_name=self.model_name, model=self.model, score=self.score)

        except Exception as e:
            model_finder = ModelFinderException(
                "Failed in [{0}] class [{1}] method [{2}]"
                    .format(self.__module__, ModelFinder.__name__,
                            self.get_best_model_fraud_detection.__name__))
            raise Exception(model_finder.error_message_detail(str(e), sys)) from e

    def get_best_params_for_naive_bayes_credit_defaulter(self, train_x, train_y):
        """
        Method Name: get_best_params_for_naive_bayes
        Description: get the parameters for the Naive Bayes's Algorithm which give the best accuracy.
        Use Hyper Parameter Tuning.
        Output: The model with the best parameters
        On Failure: Raise Exception

        Written By: iNeuron Intelligence
        Version: 1.0
        Revisions: None

        """
        try:
            self.logger_object.log('Entered the get_best_params_for_naive_bayes method of the Model_Finder class')

            # initializing with different combination of parameters
            param_grid = {"var_smoothing": [1e-9, 0.1, 0.001, 0.5, 0.05, 0.01, 1e-8, 1e-7, 1e-6, 1e-10, 1e-11]}

            # Creating an object of the Grid Search class
            grid = GridSearchCV(estimator=GaussianNB(), param_grid=param_grid, cv=3, verbose=3)
            # finding the best parameters
            grid.fit(train_x, train_y)

            # extracting the best parameters
            var_smoothing = grid.best_params_['var_smoothing']

            # creating a new model with the best parameters
            gnb = GaussianNB(var_smoothing=var_smoothing)
            # training the mew model
            gnb.fit(train_x, train_y)
            self.logger_object.log('Naive Bayes best params: ' + str(
                grid.best_params_) + '. Exited the get_best_params_for_naive_bayes method of the Model_Finder class')
            return gnb
        except Exception as e:
            model_finder = ModelFinderException(
                "Failed in [{0}] class [{1}] method [{2}]"
                    .format(self.__module__, ModelFinder.__name__,
                            self.get_best_params_for_naive_bayes_credit_defaulter.__name__))
            raise Exception(model_finder.error_message_detail(str(e), sys)) from e

    def get_best_params_for_xgboost_credit_defaulter(self, train_x, train_y):

        """
        Method Name: get_best_params_for_xgboost
        Description: get the parameters for XGBoost Algorithm which give the best accuracy.
        Use Hyper Parameter Tuning.
        Output: The model with the best parameters
        On Failure: Raise Exception

        Written By: iNeuron Intelligence
        Version: 1.0
        Revisions: None

        """
        try:
            self.logger_object.log('Entered the get_best_params_for_xgboost method of the Model_Finder class')
            # initializing with different combination of parameters
            param_grid_xgboost = {

                "n_estimators": [50, 100, 130],
                "max_depth": range(3, 11, 1),
                "random_state": [0, 50, 100]

            }
            # Creating an object of the Grid Search class
            grid = GridSearchCV(XGBClassifier(objective='binary:logistic'), param_grid_xgboost, verbose=3,
                                cv=2, n_jobs=-1)
            # finding the best parameters
            grid.fit(train_x, train_y)

            # extracting the best parameters
            random_state = grid.best_params_['random_state']
            max_depth = grid.best_params_['max_depth']
            n_estimators = grid.best_params_['n_estimators']

            # creating a new model with the best parameters
            xgb = XGBClassifier(random_state=random_state, max_depth=max_depth,
                                n_estimators=n_estimators, n_jobs=-1)
            # training the mew model
            xgb.fit(train_x, train_y)
            self.logger_object.log('XGBoost best params: ' + str(
                grid.best_params_) + '. Exited the get_best_params_for_xgboost method of the Model_Finder class')
            return xgb
        except Exception as e:
            model_finder = ModelFinderException(
                "Failed in [{0}] class [{1}] method [{2}]"
                    .format(self.__module__, ModelFinder.__name__,
                            self.get_best_params_for_xgboost_credit_defaulter.__name__))
            raise Exception(model_finder.error_message_detail(str(e), sys)) from e

    def get_best_model_credit_deaulter(self, train_x, train_y, test_x, test_y, cluster_no):
        """
        Method Name: get_best_model
        Description: Find out the Model which has the best AUC score.
        Output: The best model name and the model object
        On Failure: Raise Exception

        Written By: iNeuron Intelligence
        Version: 1.0
        Revisions: None

        """
        # create best model for XGBoost
        try:
            self.logger_object.log('Entered the get_best_model method of the Model_Finder class')

            title_generator = " Cluster " + cluster_no + " model {}"

            # XG Boost model

            self.model_name.append('XG_BOOST')
            title = title_generator.format('XG_BOOST')
            self.logger_object.log('Entered the get_best_model method of the Model_Finder class')
            xgboost = self.get_best_params_for_xgboost_credit_defaulter(train_x, train_y)
            prediction_xgboost = xgboost.predict(test_x)  # Predictions using the XGBoost Model

            if len(test_y.unique()) == 1:  # if there is only one label in y, then roc_auc_score returns error. We
                # will use accuracy in that case
                xgboost_score = accuracy_score(test_y, prediction_xgboost)
                self.logger_object.log('Accuracy for XGBoost:' + str(xgboost_score))  # Log AUC
            else:
                xgboost_score = roc_auc_score(test_y, prediction_xgboost)  # AUC for XGBoost
                self.logger_object.log('AUC for XGBoost:' + str(xgboost_score))  # Log AUC
                y_score = xgboost.predict_proba(test_x)[:, 1]
                fpr, tpr, thresholds = roc_curve(test_y, y_score, pos_label=test_y.unique()[1])
                AccurayGraph().save_roc_curve_plot_binary_classification(fpr, tpr, self.project_id,
                                                                         execution_id=self.logger_object.execution_id,
                                                                         file_object=self.file_object,
                                                                         title=title)

            self.model.append(xgboost)
            self.score.append(xgboost_score)

            # create best model for naive bayes
            self.model_name.append('NAIVE_BAYES')
            title = title_generator.format('NAIVE_BAYES')
            naive_bayes = self.get_best_params_for_naive_bayes_credit_defaulter(train_x, train_y)
            prediction_naive_bayes = naive_bayes.predict(test_x)  # prediction using the Random Forest Algorithm
            self.model.append(naive_bayes)
            if len(test_y.unique()) == 1:  # if there is only one label in y,
                # then roc_auc_score returns error. We will use accuracy in that case
                naive_bayes_score = accuracy_score(test_y, prediction_naive_bayes)
                self.logger_object.log('Accuracy for naive bayes score' + str(naive_bayes_score))
            else:
                naive_bayes_score = roc_auc_score(test_y, prediction_naive_bayes)  # AUC for Random Forest
                self.logger_object.log('AUC for naive bayes score:' + str(naive_bayes_score))
                y_score = naive_bayes.predict_proba(test_x)[:, 1]
                fpr, tpr, thresholds = roc_curve(test_y, y_score, pos_label=test_y.unique()[0])
                AccurayGraph().save_roc_curve_plot_binary_classification(fpr, tpr, self.project_id,
                                                                         execution_id=self.logger_object.execution_id,
                                                                         file_object=self.file_object,
                                                                         title=title)

            self.score.append(naive_bayes_score)
            # create best model for Random forest
            self.model_name.append('Random_Forest')
            title = title_generator.format('Random_Forest')
            random_forest = self.get_best_params_for_random_forest(train_x, train_y)
            prediction_random_forest = random_forest.predict(test_x)
            self.model.append(random_forest)
            if len(test_y.unique()) == 1:
                random_forest_score = accuracy_score(test_y, prediction_random_forest)
                self.logger_object.log('Accuracy for Random Forest' + str(random_forest_score))
            else:
                random_forest_score = roc_auc_score(test_y, prediction_random_forest)  # AUC for Random Forest
                self.logger_object.log('AUC for Random Forest' + str(random_forest_score))
                y_score = random_forest.predict_proba(test_x)[:, 1]
                fpr, tpr, thresholds = roc_curve(test_y, y_score, pos_label=test_y.unique()[1])
                AccurayGraph().save_roc_curve_plot_binary_classification(fpr, tpr, self.project_id,
                                                                         execution_id=self.logger_object.execution_id,
                                                                         file_object=self.file_object,
                                                                         title=title)

            self.score.append(random_forest_score)

            # create best model for KNN
            self.model_name.append('KNN')
            title = title_generator.format('KNN')
            knn_clf = self.get_best_params_for_KNN(train_x, train_y)
            prediction_knn = knn_clf.predict(test_x)
            self.model.append(knn_clf)
            if len(test_y.unique()) == 1:
                knn_score = accuracy_score(test_y, prediction_knn)
                self.logger_object.log('Accuracy for KNN clf' + str(knn_score))
            else:
                knn_score = roc_auc_score(test_y, prediction_knn)  # AUC for Random Forest
                self.logger_object.log('AUC for KNN' + str(knn_score))
                y_score = knn_clf.predict_proba(test_x)[:, 1]
                fpr, tpr, thresholds = roc_curve(test_y, y_score, pos_label=test_y.unique()[1])
                AccurayGraph().save_roc_curve_plot_binary_classification(fpr, tpr, self.project_id,
                                                                         execution_id=self.logger_object.execution_id,
                                                                         file_object=self.file_object,
                                                                         title=title)
            self.score.append(knn_score)

            if len(test_y.unique()) != 1:
                """ 5. SVC """
                self.model_name.append("SVC")
                title = title_generator.format("SVC")
                svc_clf = self.get_best_params_for_svm_phising_classifier(train_x, train_y)
                prediction_svc = svc_clf.predict(test_x)
                self.model.append(svc_clf)
                if len(test_y.unique()) == 1:
                    svc_score = accuracy_score(test_y, prediction_svc)
                    self.logger_object.log('Accuracy for svc clf' + str(svc_score))
                else:
                    svc_score = roc_auc_score(test_y, prediction_svc)  # AUC for Random Forest
                    self.logger_object.log('AUC for svc' + str(svc_score))
                    y_score = svc_clf.predict_proba(test_x)[:, 1]
                    fpr, tpr, thresholds = roc_curve(test_y, y_score, pos_label=test_y.unique()[1])
                    AccurayGraph().save_roc_curve_plot_binary_classification(fpr, tpr, self.project_id,
                                                                             execution_id=self.logger_object.execution_id,
                                                                             file_object=self.file_object,
                                                                             title=title)
                self.score.append(svc_score)

            AccurayGraph().save_accuracy_bar_graph(
                model_name_list=self.model_name,
                accuracy_score_list=self.score,
                project_id=self.project_id,
                execution_id=self.logger_object.execution_id,
                file_object=self.file_object,
                x_label="Model List",
                y_label="Accuracy score comparison {}".format(self.model_name),
                title="Cluster " + str(cluster_no) + "Accuracy Score "
            )
            execution_model_comparison_id = str(uuid.uuid4())

            for data in zip(self.model_name, self.score):
                self.save_accuracy_data(model_name=data[0], score=data[1],
                                        execution_model_comparision_id=execution_model_comparison_id)

            return self.get_best_model_on_score(model_name=self.model_name, model=self.model, score=self.score)


        except Exception as e:
            model_finder = ModelFinderException(
                "Failed in [{0}] class [{1}] method [{2}]"
                    .format(self.__module__, ModelFinder.__name__,
                            self.get_best_model_credit_deaulter.__name__))
            raise Exception(model_finder.error_message_detail(str(e), sys)) from e

    """phishing classifier"""

    def get_best_params_for_svm_phising_classifier(self, train_x, train_y):
        """
        Method Name: get_best_params_for_naive_bayes
        Description: get the parameters for the SVM Algorithm which give the best accuracy.
        Use Hyper Parameter Tuning.
        Output: The model with the best parameters
        On Failure: Raise Exception

        Written By: iNeuron Intelligence
        Version: 1.0
        Revisions: None

        """
        try:
            self.logger_object.log('Entered the get_best_params_for_svm method of the Model_Finder class')

            # initializing with different combination of parameters
            param_grid = {"kernel": ['rbf', 'sigmoid'],
                          "C": [0.1, 0.5, 1.0],
                          "random_state": [0, 100, 200, 300]}

            # Creating an object of the Grid Search class
            grid = GridSearchCV(SVC(), param_grid=param_grid, cv=5, verbose=3)
            # finding the best parameters
            grid.fit(train_x, train_y)

            # extracting the best parameters
            kernel = grid.best_params_['kernel']
            c = grid.best_params_['C']
            random_state = grid.best_params_['random_state']

            # creating a new model with the best parameters
            sv_classifier = SVC(kernel=kernel, C=c, random_state=random_state, probability=True)
            # training the mew model
            sv_classifier.fit(train_x, train_y)
            self.logger_object.log('SVM best params: ' + str(
                grid.best_params_) + '. Exited the get_best_params_for_svm method of the Model_Finder class')

            return sv_classifier
        except Exception as e:
            model_finder = ModelFinderException(
                "Failed in [{0}] class [{1}] method [{2}]"
                    .format(self.__module__, ModelFinder.__name__,
                            self.get_best_params_for_svm_phising_classifier.__name__))
            raise Exception(model_finder.error_message_detail(str(e), sys)) from e

    def get_best_params_for_xgboost_phising_classifier(self, train_x, train_y):

        """
        Method Name: get_best_params_for_xgboost
        Description: get the parameters for XGBoost Algorithm which give the best accuracy.
        Use Hyper Parameter Tuning.
        Output: The model with the best parameters
        On Failure: Raise Exception

        Written By: iNeuron Intelligence
        Version: 1.0
        Revisions: None

        """
        try:
            self.logger_object.log('Entered the get_best_params_for_xgboost method of the Model_Finder class')
            # initializing with different combination of parameters
            param_grid_xgboost = {

                "n_estimators": [100, 130], "criterion": ['gini', 'entropy'],
                "max_depth": range(8, 10, 1)

            }
            # Creating an object of the Grid Search class
            grid = GridSearchCV(XGBClassifier(objective='binary:logistic'), param_grid_xgboost, verbose=3,
                                cv=5)
            # finding the best parameters
            grid.fit(train_x, train_y)

            # extracting the best parameters
            criterion = grid.best_params_['criterion']
            max_depth = grid.best_params_['max_depth']
            n_estimators = grid.best_params_['n_estimators']

            # creating a new model with the best parameters
            xgb = XGBClassifier(criterion=criterion, max_depth=max_depth, n_estimators=n_estimators,
                                n_jobs=-1)
            # training the mew model
            xgb.fit(train_x, train_y)
            self.logger_object.log('XGBoost best params: ' + str(
                grid.best_params_) + '. Exited the get_best_params_for_xgboost method of the Model_Finder class')
            return xgb
        except Exception as e:
            model_finder = ModelFinderException(
                "Failed in [{0}] class [{1}] method [{2}]"
                    .format(self.__module__, ModelFinder.__name__,
                            self.get_best_params_for_xgboost_phising_classifier.__name__))
            raise Exception(model_finder.error_message_detail(str(e), sys)) from e

    def get_best_model_phising_classifier(self, train_x, train_y, test_x, test_y, cluster_no):
        """
        Method Name: get_best_model
        Description: Find out the Model which has the best AUC score.
        Output: The best model name and the model object
        On Failure: Raise Exception

        Written By: iNeuron Intelligence
        Version: 1.0
        Revisions: None

        """
        # create best model for XGBoost
        try:
            self.logger_object.log('Entered the get_best_model method of the Model_Finder class')

            title_generator = " Cluster " + cluster_no + " model {}"

            # XG Boost model

            self.model_name.append('XG_BOOST')
            title = title_generator.format('XG_BOOST')
            self.logger_object.log('Entered the get_best_model method of the Model_Finder class')
            xgboost = self.get_best_params_for_xgboost_phising_classifier(train_x, train_y)
            prediction_xgboost = xgboost.predict(test_x)  # Predictions using the XGBoost Model

            if len(test_y.unique()) == 1:  # if there is only one label in y, then roc_auc_score returns error. We
                # will use accuracy in that case
                xgboost_score = accuracy_score(test_y, prediction_xgboost)
                self.logger_object.log('Accuracy for XGBoost:' + str(xgboost_score))  # Log AUC
            else:
                xgboost_score = roc_auc_score(test_y, prediction_xgboost)  # AUC for XGBoost
                self.logger_object.log('AUC for XGBoost:' + str(xgboost_score))  # Log AUC
                y_score = xgboost.predict_proba(test_x)[:, 1]
                fpr, tpr, thresholds = roc_curve(test_y, y_score, pos_label=test_y.unique()[1])
                AccurayGraph().save_roc_curve_plot_binary_classification(fpr, tpr, self.project_id,
                                                                         execution_id=self.logger_object.execution_id,
                                                                         file_object=self.file_object,
                                                                         title=title)

            self.model.append(xgboost)
            self.score.append(xgboost_score)

            # create best model for naive bayes
            self.model_name.append('NAIVE_BAYES')
            title = title_generator.format('NAIVE_BAYES')
            naive_bayes = self.get_best_params_for_naive_bayes(train_x, train_y)
            prediction_naive_bayes = naive_bayes.predict(test_x)  # prediction using the Random Forest Algorithm
            self.model.append(naive_bayes)
            if len(test_y.unique()) == 1:  # if there is only one label in y,
                # then roc_auc_score returns error. We will use accuracy in that case
                naive_bayes_score = accuracy_score(test_y, prediction_naive_bayes)
                self.logger_object.log('Accuracy for naive bayes score' + str(naive_bayes_score))
            else:
                naive_bayes_score = roc_auc_score(test_y, prediction_naive_bayes)  # AUC for Random Forest
                self.logger_object.log('AUC for naive bayes score:' + str(naive_bayes_score))
                y_score = naive_bayes.predict_proba(test_x)[:, 1]
                fpr, tpr, thresholds = roc_curve(test_y, y_score, pos_label=test_y.unique()[0])
                AccurayGraph().save_roc_curve_plot_binary_classification(fpr, tpr, self.project_id,
                                                                         execution_id=self.logger_object.execution_id,
                                                                         file_object=self.file_object,
                                                                         title=title)

            self.score.append(naive_bayes_score)
            # create best model for Random forest
            self.model_name.append('Random_Forest')
            title = title_generator.format('Random_Forest')
            random_forest = self.get_best_params_for_random_forest(train_x, train_y)
            prediction_random_forest = random_forest.predict(test_x)
            self.model.append(random_forest)
            if len(test_y.unique()) == 1:
                random_forest_score = accuracy_score(test_y, prediction_random_forest)
                self.logger_object.log('Accuracy for Random Forest' + str(random_forest_score))
            else:
                random_forest_score = roc_auc_score(test_y, prediction_random_forest)  # AUC for Random Forest
                self.logger_object.log('AUC for Random Forest' + str(random_forest_score))
                y_score = random_forest.predict_proba(test_x)[:, 1]
                fpr, tpr, thresholds = roc_curve(test_y, y_score, pos_label=test_y.unique()[1])
                AccurayGraph().save_roc_curve_plot_binary_classification(fpr, tpr, self.project_id,
                                                                         execution_id=self.logger_object.execution_id,
                                                                         file_object=self.file_object,
                                                                         title=title)

            self.score.append(random_forest_score)

            # create best model for KNN
            self.model_name.append('KNN')
            title = title_generator.format('KNN')
            knn_clf = self.get_best_params_for_KNN(train_x, train_y)
            prediction_knn = knn_clf.predict(test_x)
            self.model.append(knn_clf)
            if len(test_y.unique()) == 1:
                knn_score = accuracy_score(test_y, prediction_knn)
                self.logger_object.log('Accuracy for KNN clf' + str(knn_score))
            else:
                knn_score = roc_auc_score(test_y, prediction_knn)  # AUC for Random Forest
                self.logger_object.log('AUC for KNN' + str(knn_score))
                y_score = knn_clf.predict_proba(test_x)[:, 1]
                fpr, tpr, thresholds = roc_curve(test_y, y_score, pos_label=test_y.unique()[1])
                AccurayGraph().save_roc_curve_plot_binary_classification(fpr, tpr, self.project_id,
                                                                         execution_id=self.logger_object.execution_id,
                                                                         file_object=self.file_object,
                                                                         title=title)
            self.score.append(knn_score)

            if len(test_y.unique()) != 1:
                """ 5. SVC """
                self.model_name.append("SVC")
                title = title_generator.format("SVC")
                svc_clf = self.get_best_params_for_svm_phising_classifier(train_x, train_y)
                prediction_svc = svc_clf.predict(test_x)
                self.model.append(svc_clf)
                if len(test_y.unique()) == 1:
                    svc_score = accuracy_score(test_y, prediction_svc)
                    self.logger_object.log('Accuracy for svc clf' + str(svc_score))
                else:
                    svc_score = roc_auc_score(test_y, prediction_svc)  # AUC for Random Forest
                    self.logger_object.log('AUC for svc' + str(svc_score))
                    y_score = svc_clf.predict_proba(test_x)[:, 1]
                    fpr, tpr, thresholds = roc_curve(test_y, y_score, pos_label=test_y.unique()[1])
                    AccurayGraph().save_roc_curve_plot_binary_classification(fpr, tpr, self.project_id,
                                                                             execution_id=self.logger_object.execution_id,
                                                                             file_object=self.file_object,
                                                                             title=title)
                self.score.append(svc_score)

            AccurayGraph().save_accuracy_bar_graph(
                model_name_list=self.model_name,
                accuracy_score_list=self.score,
                project_id=self.project_id,
                execution_id=self.logger_object.execution_id,
                file_object=self.file_object,
                x_label="Model List",
                y_label="Accuracy score comparison {}".format(self.model_name),
                title="Cluster " + str(cluster_no) + "Accuracy Score "
            )
            execution_model_comparison_id = str(uuid.uuid4())

            for data in zip(self.model_name, self.score):
                self.save_accuracy_data(model_name=data[0], score=data[1],
                                        execution_model_comparision_id=execution_model_comparison_id)

            return self.get_best_model_on_score(model_name=self.model_name, model=self.model, score=self.score)

        except Exception as e:
            model_finder = ModelFinderException(
                "Failed in [{0}] class [{1}] method [{2}]"
                    .format(self.__module__, ModelFinder.__name__,
                            self.get_best_model_phising_classifier.__name__))
            raise Exception(model_finder.error_message_detail(str(e), sys)) from e

    """Forest cover classifier """

    def get_best_params_for_random_forest_forest_cover_clf(self, train_x, train_y):
        """
        Method Name: get_best_params_for_random_forest
        Description: get the parameters for Random Forest Algorithm which give the best accuracy.
        Use Hyper Parameter Tuning.
        Output: The model with the best parameters
        On Failure: Raise Exception

        Written By: iNeuron Intelligence
        Version: 1.0
        Revisions: None

        """

        try:
            self.logger_object.log('Entered the get_best_params_for_random_forest method of the Model_Finder class')
            # initializing with different combination of parameters
            param_grid = {"n_estimators": [10, 50, 100, 130], "criterion": ['gini', 'entropy'],
                          "max_depth": range(2, 4, 1), "max_features": ['auto', 'log2']}

            # Creating an object of the Grid Search class
            grid = GridSearchCV(estimator=RandomForestClassifier(), param_grid=param_grid, cv=5, verbose=3, n_jobs=-1)
            # finding the best parameters
            grid.fit(train_x, train_y)

            # extracting the best parameters
            criterion = grid.best_params_['criterion']
            max_depth = grid.best_params_['max_depth']
            max_features = grid.best_params_['max_features']
            n_estimators = grid.best_params_['n_estimators']

            # creating a new model with the best parameters
            clf = RandomForestClassifier(n_estimators=n_estimators, criterion=criterion,
                                         max_depth=max_depth, max_features=max_features)
            # training the mew model
            clf.fit(train_x, train_y)
            self.logger_object.log('Random Forest best params: ' + str(
                grid.best_params_) + '. Exited the get_best_params_for_random_forest method of the Model_Finder class')

            return clf
        except Exception as e:
            model_finder = ModelFinderException(
                "Failed in [{0}] class [{1}] method [{2}]"
                    .format(self.__module__, ModelFinder.__name__,
                            self.get_best_params_for_random_forest_forest_cover_clf.__name__))
            raise Exception(model_finder.error_message_detail(str(e), sys)) from e

    def get_best_params_for_xgboost_forest_cover_clf(self, train_x, train_y):

        """
        Method Name: get_best_params_for_xgboost
        Description: get the parameters for XGBoost Algorithm which give the best accuracy.
        Use Hyper Parameter Tuning.
        Output: The model with the best parameters
        On Failure: Raise Exception

        Written By: iNeuron Intelligence
        Version: 1.0
        Revisions: None

        """

        try:
            self.logger_object.log('Entered the get_best_params_for_xgboost method of the Model_Finder class')
            # initializing with different combination of parameters
            param_grid_xgboost = {

                'learning_rate': [0.5, 0.1, 0.01, 0.001],
                'max_depth': [3, 5, 10, 20],
                'n_estimators': [10, 50, 100, 200]

            }
            # Creating an object of the Grid Search class
            grid = GridSearchCV(XGBClassifier(objective='multi:softprob'), param_grid_xgboost, verbose=3, cv=5,
                                n_jobs=-1)
            # finding the best parameters
            grid.fit(train_x, train_y)

            # extracting the best parameters
            learning_rate = grid.best_params_['learning_rate']
            max_depth = grid.best_params_['max_depth']
            n_estimators = grid.best_params_['n_estimators']

            # creating a new model with the best parameters
            xgb = XGBClassifier(learning_rate=learning_rate, max_depth=max_depth, n_estimators=n_estimators)
            # training the mew model
            xgb.fit(train_x, train_y)
            self.logger_object.log('XGBoost best params: ' + str(
                grid.best_params_) + '. Exited the get_best_params_for_xgboost method of the Model_Finder class')
            return xgb
        except Exception as e:
            model_finder = ModelFinderException(
                "Failed in [{0}] class [{1}] method [{2}]"
                    .format(self.__module__, ModelFinder.__name__,
                            self.get_best_params_for_xgboost_forest_cover_clf.__name__))
            raise Exception(model_finder.error_message_detail(str(e), sys)) from e

    def get_best_model_forest_cover(self, train_x, train_y, test_x, test_y, cluster_no=None):
        """
        Method Name: get_best_model
        Description: Find out the Model which has the best AUC score.
        Output: The best model name and the model object
        On Failure: Raise Exception

        Written By: iNeuron Intelligence
        Version: 1.0
        Revisions: None

        """

        # create best model for XGBoost
        try:
            self.logger_object.log('Entered the get_best_model method of the Model_Finder class')
            if cluster_no is not None:
                title_generator = " Cluster " + cluster_no + " model {}"
            else:
                title_generator = "Model {}"
            # XG Boost model
            self.model_name.append('XG_BOOST')
            title = title_generator.format('XG_BOOST')
            self.logger_object.log('Entered the get_best_model method of the Model_Finder class')
            xgboost = self.get_best_params_for_xgboost(train_x, train_y)
            prediction_xgboost = xgboost.predict(test_x)  # Predictions using the XGBoost Model

            if len(test_y.unique()) == 1:  # if there is only one label in y, then roc_auc_score returns error. We
                # will use accuracy in that case
                xgboost_score = accuracy_score(test_y, prediction_xgboost)
                self.logger_object.log('Accuracy for XGBoost:' + str(xgboost_score))  # Log AUC
            else:
                y_scores = xgboost.predict_proba(test_x)
                AccurayGraph().save_plot_multiclass_roc_curve(test_y, y_scores, xgboost,
                                                              project_id=self.project_id,
                                                              execution_id=self.logger_object.execution_id,
                                                              file_object=self.file_object,
                                                              title="XGBoost ROC curve"
                                                              )
                xgboost_score = roc_auc_score(test_y, y_scores, multi_class='ovr')  # AUC for XGBoost
                self.logger_object.log('AUC for XGBoost:' + str(xgboost_score))  # Log AUC

            self.model.append(xgboost)
            self.score.append(xgboost_score)

            # create best model for naive bayes
            self.model_name.append('NAIVE_BAYES')
            title = title_generator.format('NAIVE_BAYES')
            naive_bayes = self.get_best_params_for_naive_bayes(train_x, train_y)
            prediction_naive_bayes = naive_bayes.predict(test_x)  # prediction using the Random Forest Algorithm
            self.model.append(naive_bayes)
            if len(test_y.unique()) == 1:  # if there is only one label in y,
                # then roc_auc_score returns error. We will use accuracy in that case
                naive_bayes_score = accuracy_score(test_y, prediction_naive_bayes)
                self.logger_object.log('Accuracy for naive bayes score' + str(naive_bayes_score))
            else:
                y_scores = naive_bayes.predict_proba(test_x)
                AccurayGraph().save_plot_multiclass_roc_curve(test_y, y_scores, naive_bayes,
                                                              project_id=self.project_id,
                                                              execution_id=self.logger_object.execution_id,
                                                              file_object=self.file_object,
                                                              title=title + self.model_name[-1]
                                                              )
                naive_bayes_score = roc_auc_score(test_y, y_scores,
                                                  multi_class='ovr')  # AUC for Random Forest
                self.logger_object.log('AUC for naive bayes score:' + str(naive_bayes_score))

            self.score.append(naive_bayes_score)
            # create best model for Random forest
            self.model_name.append('Random_Forest')
            title = title_generator.format('Random_Forest')
            random_forest = self.get_best_params_for_random_forest(train_x, train_y)
            prediction_random_forest = random_forest.predict(test_x)
            self.model.append(random_forest)
            if len(test_y.unique()) == 1:
                random_forest_score = accuracy_score(test_y, prediction_random_forest)
                self.logger_object.log('Accuracy for Random Forest' + str(random_forest_score))
            else:
                y_scores = random_forest.predict_proba(test_x)
                AccurayGraph().save_plot_multiclass_roc_curve(test_y, y_scores, random_forest,
                                                              project_id=self.project_id,
                                                              execution_id=self.logger_object.execution_id,
                                                              file_object=self.file_object,
                                                              title=title + self.model_name[-1]
                                                              )
                random_forest_score = roc_auc_score(test_y, y_scores,
                                                    multi_class='ovr')  # AUC for Random Forest
                self.logger_object.log('AUC for Random Forest' + str(random_forest_score))

            self.score.append(random_forest_score)

            # create best model for KNN
            self.model_name.append('KNN')
            title = title_generator.format('KNN')
            knn_clf = self.get_best_params_for_KNN(train_x, train_y)
            prediction_knn = knn_clf.predict(test_x)
            self.model.append(knn_clf)
            if len(test_y.unique()) == 1:
                knn_score = accuracy_score(test_y, prediction_knn)
                self.logger_object.log('Accuracy for KNN clf' + str(knn_score))
            else:
                y_scores = knn_clf.predict_proba(test_x)
                AccurayGraph().save_plot_multiclass_roc_curve(test_y, y_scores, knn_clf,
                                                              project_id=self.project_id,
                                                              execution_id=self.logger_object.execution_id,
                                                              file_object=self.file_object,
                                                              title=title + self.model_name[-1]
                                                              )
                knn_score = roc_auc_score(test_y, y_scores, multi_class='ovr')  # AUC for Random Forest
                self.logger_object.log('AUC for KNN' + str(knn_score))

            self.score.append(knn_score)

            """ 5. SVC """
            if len(test_y.unique()) != 1:
                self.model_name.append("SVC")
                title = title_generator.format("SVC")
                svc_clf = self.get_best_params_for_svm_fraud_detection_and_scania(train_x, train_y)
                prediction_svc = svc_clf.predict(test_x)
                self.model.append(svc_clf)
                if len(test_y.unique()) == 1:
                    svc_score = accuracy_score(test_y, prediction_svc)
                    self.logger_object.log('Accuracy for svc clf' + str(svc_score))
                else:
                    y_scores = svc_clf.predict_proba(test_x)
                    AccurayGraph().save_plot_multiclass_roc_curve(test_y, y_scores, svc_clf,
                                                                  project_id=self.project_id,
                                                                  execution_id=self.logger_object.execution_id,
                                                                  file_object=self.file_object,
                                                                  title=title + self.model_name[-1]
                                                                  )
                    svc_score = roc_auc_score(test_y, y_scores, multi_class='ovr')  # AUC for Random Forest
                    self.logger_object.log('AUC for svc' + str(svc_score))

                self.score.append(svc_score)

            AccurayGraph().save_accuracy_bar_graph(
                model_name_list=self.model_name,
                accuracy_score_list=self.score,
                project_id=self.project_id,
                execution_id=self.logger_object.execution_id,
                file_object=self.file_object,
                x_label="Model List",
                y_label="Accuracy score comparison {}".format(self.model_name),
                title="Accuracy Score "
            )
            execution_model_comparison_id = str(uuid.uuid4())

            for data in zip(self.model_name, self.score):
                self.save_accuracy_data(model_name=data[0], score=data[1],
                                        execution_model_comparision_id=execution_model_comparison_id)

            # comparing the two models
            return self.get_best_model_on_score(model_name=self.model_name, model=self.model, score=self.score)

        except Exception as e:
            model_finder = ModelFinderException(
                "Failed in [{0}] class [{1}] method [{2}]"
                    .format(self.__module__, ModelFinder.__name__,
                            self.get_best_model_forest_cover.__name__))
            raise Exception(model_finder.error_message_detail(str(e), sys)) from e

    def get_best_model_scania_truck(self, train_x, train_y, test_x, test_y, cluster_no=None):
        """
        Method Name: get_best_model
        Description: Find out the Model which has the best AUC score.
        Output: The best model name and the model object
        On Failure: Raise Exception

        Written By: iNeuron Intelligence
        Version: 1.0
        Revisions: None

        """
        # create best model for XGBoost
        try:
            self.logger_object.log('Entered the get_best_model method of the Model_Finder class')
            if cluster_no is not None:
                title_generator = " Cluster " + cluster_no + " model {}"
            else:
                title_generator = "Model {}"
            # XG Boost model
            self.model_name.append('XG_BOOST')
            title = title_generator.format('XG_BOOST')

            xgboost = self.get_best_params_for_xgboost(train_x, train_y)
            prediction_xgboost = xgboost.predict(test_x)  # Predictions using the XGBoost Model

            if len(test_y.unique()) == 1:  # if there is only one label in y, then roc_auc_score returns error. We
                # will use accuracy in that case
                xgboost_score = accuracy_score(test_y, prediction_xgboost)
                self.logger_object.log('Accuracy for XGBoost:' + str(xgboost_score))  # Log AUC
            else:
                xgboost_score = roc_auc_score(test_y, prediction_xgboost)  # AUC for XGBoost
                self.logger_object.log('AUC for XGBoost:' + str(xgboost_score))  # Log AUC
                y_score = xgboost.predict_proba(test_x)[:, 1]
                fpr, tpr, thresholds = roc_curve(test_y, y_score, pos_label=test_y.unique()[1])
                AccurayGraph().save_roc_curve_plot_binary_classification(fpr, tpr, self.project_id,
                                                                         execution_id=self.logger_object.execution_id,
                                                                         file_object=self.file_object,
                                                                         title=title)

            self.model.append(xgboost)
            self.score.append(xgboost_score)
            """
            # create best model for naive bayes
            self.model_name.append('NAIVE_BAYES')
            title = title_generator.format('NAIVE_BAYES')
            naive_bayes = self.get_best_params_for_naive_bayes(train_x, train_y)
            prediction_naive_bayes = naive_bayes.predict(test_x)  # prediction using the Random Forest Algorithm
            self.model.append(naive_bayes)
            if len(test_y.unique()) == 1:  # if there is only one label in y,
                # then roc_auc_score returns error. We will use accuracy in that case
                naive_bayes_score = accuracy_score(test_y, prediction_naive_bayes)
                self.logger_object.log('Accuracy for naive bayes score' + str(naive_bayes_score))
            else:
                naive_bayes_score = roc_auc_score(test_y, prediction_naive_bayes)  # AUC for Random Forest
                self.logger_object.log('AUC for naive bayes score:' + str(naive_bayes_score))
                y_score = naive_bayes.predict_proba(test_x)[:, 1]
                fpr, tpr, thresholds = roc_curve(test_y, y_score, pos_label=test_y.unique()[0])
                AccurayGraph().save_roc_curve_plot_binary_classification(fpr, tpr, self.project_id,
                                                                         execution_id=self.logger_object.execution_id,
                                                                         file_object=self.file_object,
                                                                         title=title)

            self.score.append(naive_bayes_score)
            """
            # create best model for Random forest
            self.model_name.append('Random_Forest')
            title = title_generator.format('Random_Forest')
            random_forest = self.get_best_params_for_random_forest(train_x, train_y)
            prediction_random_forest = random_forest.predict(test_x)
            self.model.append(random_forest)
            if len(test_y.unique()) == 1:
                random_forest_score = accuracy_score(test_y, prediction_random_forest)
                self.logger_object.log('Accuracy for Random Forest' + str(random_forest_score))
            else:
                random_forest_score = roc_auc_score(test_y, prediction_random_forest)  # AUC for Random Forest
                self.logger_object.log('AUC for Random Forest' + str(random_forest_score))
                y_score = random_forest.predict_proba(test_x)[:, 1]
                fpr, tpr, thresholds = roc_curve(test_y, y_score, pos_label=test_y.unique()[1])
                AccurayGraph().save_roc_curve_plot_binary_classification(fpr, tpr, self.project_id,
                                                                         execution_id=self.logger_object.execution_id,
                                                                         file_object=self.file_object,
                                                                         title=title)

            self.score.append(random_forest_score)
            """
            # create best model for KNN
            self.model_name.append('KNN')
            title = title_generator.format('KNN')
            knn_clf = self.get_best_params_for_KNN(train_x, train_y)
            prediction_knn = knn_clf.predict(test_x)
            self.model.append(knn_clf)
            if len(test_y.unique()) == 1:
                knn_score = accuracy_score(test_y, prediction_knn)
                self.logger_object.log('Accuracy for KNN clf' + str(knn_score))
            else:
                knn_score = roc_auc_score(test_y, prediction_knn)  # AUC for Random Forest
                self.logger_object.log('AUC for KNN' + str(knn_score))
                y_score = knn_clf.predict_proba(test_x)[:, 1]
                fpr, tpr, thresholds = roc_curve(test_y, y_score, pos_label=test_y.unique()[1])
                AccurayGraph().save_roc_curve_plot_binary_classification(fpr, tpr, self.project_id,
                                                                         execution_id=self.logger_object.execution_id,
                                                                         file_object=self.file_object,
                                                                         title=title)
            self.score.append(knn_score)
            
             5. SVC 
            if len(test_y.unique()) != 1:
                self.model_name.append("SVC")
                title = title_generator.format("SVC")
                svc_clf = self.get_best_params_for_svm_fraud_detection_and_scania(train_x, train_y)
                prediction_svc = svc_clf.predict(test_x)
                self.model.append(svc_clf)
                if len(test_y.unique()) == 1:
                    svc_score = accuracy_score(test_y, prediction_svc)
                    self.logger_object.log('Accuracy for svc clf' + str(svc_score))
                else:
                    svc_score = roc_auc_score(test_y, prediction_svc)  # AUC for Random Forest
                    self.logger_object.log('AUC for svc' + str(svc_score))
                    y_score = svc_clf.predict_proba(test_x)[:, 1]
                    fpr, tpr, thresholds = roc_curve(test_y, y_score, pos_label=test_y.unique()[1])
                    AccurayGraph().save_roc_curve_plot_binary_classification(fpr, tpr, self.project_id,
                                                                             execution_id=self.logger_object.execution_id,
                                                                             file_object=self.file_object,
                                                                             title=title)
                self.score.append(svc_score)
            """
            AccurayGraph().save_accuracy_bar_graph(
                model_name_list=self.model_name,
                accuracy_score_list=self.score,
                project_id=self.project_id,
                execution_id=self.logger_object.execution_id,
                file_object=self.file_object,
                x_label="Model List",
                y_label="Accuracy score comparison {}".format(self.model_name),
                title="Accuracy Score "
            )
            execution_model_comparison_id = str(uuid.uuid4())
            for data in zip(self.model_name, self.score):
                self.save_accuracy_data(model_name=data[0], score=data[1],
                                        execution_model_comparision_id=execution_model_comparison_id)
            # comparing the two models
            return self.get_best_model_on_score(model_name=self.model_name, model=self.model, score=self.score)
        except Exception as e:
            model_finder = ModelFinderException(
                "Failed in [{0}] class [{1}] method [{2}]"
                    .format(self.__module__, ModelFinder.__name__,
                            self.get_best_model_forest_cover.__name__))
            raise Exception(model_finder.error_message_detail(str(e), sys)) from e

    def get_best_params_for_Random_Forest_Regressor(self, train_x, train_y):
        """
        Method Name: get_best_params_for_Random_Forest_Regressor
        Description: get the parameters for Random_Forest_Regressor Algorithm which give the best accuracy.
        Use Hyper Parameter Tuning.
        Output: The model with the best parameters
        On Failure: Raise Exception

        Written By: iNeuron Intelligence
        Version: 1.0
        Revisions: None

        """
        try:
            self.logger_object.log('Entered the RandomForestReg method of the Model_Finder class')
            # initializing with different combination of parameters
            param_grid_random_forest_tree = {
                "n_estimators": [10, 20, 30],
                "max_features": ["auto", "sqrt", "log2"],
                "min_samples_split": [2, 4, 8],
                "bootstrap": [True, False]
            }

            # Creating an object of the Grid Search class
            grid = GridSearchCV(RandomForestRegressor(), param_grid_random_forest_tree, verbose=3, cv=5)
            # finding the best parameters
            grid.fit(train_x, train_y)

            # extracting the best parameters
            n_estimators = grid.best_params_['n_estimators']
            max_features = grid.best_params_['max_features']
            min_samples_split = grid.best_params_['min_samples_split']
            bootstrap = grid.best_params_['bootstrap']

            # creating a new model with the best parameters
            random_forest_reg = RandomForestRegressor(n_estimators=n_estimators,
                                                      max_features=max_features,
                                                      min_samples_split=min_samples_split,
                                                      bootstrap=bootstrap)
            # training the mew models
            random_forest_reg.fit(train_x, train_y)
            self.logger_object.log('RandomForestReg best params: ' + str(
                grid.best_params_) + '. Exited the RandomForestReg method of the Model_Finder class')
            return random_forest_reg
        except Exception as e:
            model_finder = ModelFinderException(
                "Failed in [{0}] class [{1}] method [{2}]"
                    .format(self.__module__, ModelFinder.__name__,
                            self.get_best_params_for_Random_Forest_Regressor.__name__))
            raise Exception(model_finder.error_message_detail(str(e), sys)) from e

    def get_best_params_for_linearReg(self, train_x, train_y):

        """
        Method Name: get_best_params_for_linearReg
        Description: get the parameters for LinearReg Algorithm which give the best accuracy.
        Use Hyper Parameter Tuning.
        Output: The model with the best parameters
        On Failure: Raise Exception

        Written By: iNeuron Intelligence
        Version: 1.0
        Revisions: None

        """
        try:
            self.logger_object.log('Entered the get_best_params_for_linearReg method of the Model_Finder class')
            # initializing with different combination of parameters
            param_grid_linear_reg = {
                'fit_intercept': [True, False], 'normalize': [True, False], 'copy_X': [True, False]

            }
            # Creating an object of the Grid Search class
            grid = GridSearchCV(LinearRegression(), param_grid_linear_reg, verbose=3, cv=5)
            # finding the best parameters
            grid.fit(train_x, train_y)

            # extracting the best parameters
            fit_intercept = grid.best_params_['fit_intercept']
            normalize = grid.best_params_['normalize']
            copy_x = grid.best_params_['copy_X']

            # creating a new model with the best parameters
            lin_reg = LinearRegression(fit_intercept=fit_intercept, normalize=normalize,
                                       copy_X=copy_x)
            # training the mew model
            lin_reg.fit(train_x, train_y)
            self.logger_object.log('LinearRegression best params: ' + str(
                grid.best_params_) + '. Exited the get_best_params_for_linearReg method of the Model_Finder class')
            return lin_reg
        except Exception as e:
            model_finder = ModelFinderException(
                "Failed in [{0}] class [{1}] method [{2}]"
                    .format(self.__module__, ModelFinder.__name__,
                            self.get_best_params_for_linearReg.__name__))
            raise Exception(model_finder.error_message_detail(str(e), sys)) from e

    def get_best_model_for_reg(self, train_x, train_y, test_x, test_y, cluster_no=None):
        """
        Method Name: get_best_model
        Description: Find out the Model which has the best AUC score.
        Output: The best model name and the model object
        On Failure: Raise Exception

        Written By: iNeuron Intelligence
        Version: 1.0
        Revisions: None

        """
        try:

            self.logger_object.log('Entered the get_best_model method of the Model_Finder class')
            title = "Cluster {} ".format(cluster_no) if cluster_no is not None else ''

            # Linear Regression Training

            self.model_name.append("Linear_Regression")
            linear_reg = self.get_best_params_for_linearReg(train_x, train_y)
            prediction_linear_reg = linear_reg.predict(test_x)  # Predictions using the LinearReg Model
            linear_reg_error = r2_score(test_y, prediction_linear_reg)
            self.model.append(linear_reg)
            self.score.append(linear_reg_error)

            # Decision Tree training
            self.model_name.append('Decision_Tree')
            decision_tree_reg = self.get_best_params_for_decision_tree_regressor(train_x, train_y)

            self.model.append(decision_tree_reg)
            prediction_decision_tree_reg = decision_tree_reg.predict(
                test_x)  # Predictions using the decisionTreeReg Model
            decision_tree_reg_error = r2_score(test_y, prediction_decision_tree_reg)

            self.score.append(decision_tree_reg_error)
            self.logger_object.log("Decision tree regression r2 score {}".format(decision_tree_reg_error))

            # create best model for XGBoost
            self.model_name.append('XG_BOOST')
            xgboost = self.get_best_params_for_xgboost_regressor(train_x, train_y)
            prediction_xgboost = xgboost.predict(test_x)  # Predictions using the XGBoost Model

            self.model.append(xgboost)
            prediction_xgboost_error = r2_score(test_y, prediction_xgboost)
            self.logger_object.log("XGBoost regression r2 score {}".format(prediction_xgboost_error))
            self.score.append(prediction_xgboost_error)

            self.model_name.append("Random_Forest")
            random_forest_reg = self.get_best_params_for_Random_Forest_Regressor(train_x, train_y)
            self.model.append(random_forest_reg)
            prediction_random_forest_reg = random_forest_reg.predict(test_x)
            prediction_random_forest_error = r2_score(test_y, prediction_random_forest_reg)
            self.score.append(prediction_random_forest_error)
            self.logger_object.log("Random Forest regression r2 score {}".format(prediction_random_forest_error))

            self.model_name.append("SVR")
            sv_reg = self.get_best_params_for_support_vector_regressor(train_x, train_y)
            self.model.append(sv_reg)
            prediction_sv_reg = sv_reg.predict(test_x)
            prediction_sv_reg_error = r2_score(test_y, prediction_sv_reg)
            self.score.append(prediction_sv_reg_error)
            self.logger_object.log("Support vector regression r2 score {}".format(prediction_sv_reg_error))

            """
            Visualization begin based on above model
            """
            prediction_value = [prediction_linear_reg,
                                prediction_decision_tree_reg,
                                prediction_xgboost,
                                prediction_random_forest_reg,
                                prediction_sv_reg]

            for data in zip(self.model_name, prediction_value):
                AccurayGraph().save_scatter_plot(x_axis_data=test_y, y_axis_data=data[1],
                                                 project_id=self.project_id,
                                                 execution_id=self.logger_object.execution_id,
                                                 file_object=self.file_object,
                                                 x_label="True Target values", y_label="Predicted Target value",
                                                 title=title + "Predicted vs True " + data[0])

                AccurayGraph().save_distribution_plot(data=numpy.abs(test_y - data[1]),
                                                      label="Residual distribution plot",
                                                      project_id=self.project_id,
                                                      execution_id=self.logger_object.execution_id,
                                                      file_object=self.file_object,
                                                      x_label="Error ",
                                                      y_label="frequency or occurance",
                                                      title=title + "{} residual distribution plot".format(data[0])
                                                      )

            mean_abs_error = []
            for data in prediction_value:
                mean_abs_error.append(numpy.mean(numpy.abs(test_y - data)))

            AccurayGraph().save_accuracy_bar_graph(
                model_name_list=self.model_name,
                accuracy_score_list=mean_abs_error,
                project_id=self.project_id,
                execution_id=self.logger_object.execution_id,
                file_object=self.file_object,
                x_label="Model List",
                y_label="MAE comparison between {}".format(self.model_name),
                title=title + "Mean Absolute error "
            )
            # saving accuracy data based on model on mongo db
            execution_model_comparison_id = str(uuid.uuid4())
            for data in zip(self.model_name, self.score):
                self.save_accuracy_data(model_name=data[0], score=data[1],
                                        execution_model_comparision_id=execution_model_comparison_id)
            return self.get_best_model_on_score(model_name=self.model_name, model=self.model, score=self.score)

        except Exception as e:
            model_finder = ModelFinderException(
                "Failed in [{0}] class [{1}] method [{2}]"
                    .format(self.__module__, ModelFinder.__name__,
                            self.get_best_model_for_reg.__name__))
            raise Exception(model_finder.error_message_detail(str(e), sys)) from e

    def get_best_params_for_decision_tree_regressor(self, train_x, train_y):
        """
        Method Name: get_best_params_for_DecisionTreeRegressor
        Description: get the parameters for DecisionTreeRegressor Algorithm which give the best accuracy.
        Use Hyper Parameter Tuning.
        Output: The model with the best parameters
        On Failure: Raise Exception

        Written By: iNeuron Intelligence
        Version: 1.0
        Revisions: None

        """
        try:
            self.logger_object.log(
                'Entered the get_best_params_for_DecisionTreeRegressor method of the Model_Finder class')
            # initializing with different combination of parameters
            param_grid_decision_tree = {"criterion": ["mse", "friedman_mse", "mae"],
                                        "splitter": ["best", "random"],
                                        "max_features": ["auto", "sqrt", "log2"],
                                        'max_depth': range(2, 16, 2),
                                        'min_samples_split': range(2, 16, 2)
                                        }

            # Creating an object of the Grid Search class
            grid = GridSearchCV(DecisionTreeRegressor(), param_grid_decision_tree, verbose=3, cv=5)
            # finding the best parameters
            grid.fit(train_x, train_y)

            # extracting the best parameters
            criterion = grid.best_params_['criterion']
            splitter = grid.best_params_['splitter']
            max_features = grid.best_params_['max_features']
            max_depth = grid.best_params_['max_depth']
            min_samples_split = grid.best_params_['min_samples_split']

            # creating a new model with the best parameters
            decision_tree_reg = DecisionTreeRegressor(criterion=criterion, splitter=splitter,
                                                      max_features=max_features, max_depth=max_depth,
                                                      min_samples_split=min_samples_split)
            # training the mew models
            decision_tree_reg.fit(train_x, train_y)
            self.logger_object.log('Decision Tree repressor ' + str(
                grid.best_params_) + '. exited decision tree the Model_Finder class')
            return decision_tree_reg
        except Exception as e:
            model_finder = ModelFinderException(
                "Failed in [{0}] class [{1}] method [{2}]"
                    .format(self.__module__, ModelFinder.__name__,
                            self.get_best_params_for_decision_tree_regressor.__name__))
            raise Exception(model_finder.error_message_detail(str(e), sys)) from e

    def get_best_params_for_xgboost_regressor(self, train_x, train_y):

        """
        Method Name: get_best_params_for_xgboost
        Description: get the parameters for XGBoost Algorithm which give the best accuracy.
        Use Hyper Parameter Tuning.
        Output: The model with the best parameters
        On Failure: Raise Exception

        Written By: iNeuron Intelligence
        Version: 1.0
        Revisions: None

        """
        try:
            self.logger_object.log('Entered the get_best_params_for_xgboost method of the Model_Finder class')
            # initializing with different combination of parameters
            param_grid_xgboost = {

                'learning_rate': [0.5, 0.1, 0.01, 0.001],
                'max_depth': [3, 5, 10, 20],
                'n_estimators': [10, 50, 100, 200]

            }
            # Creating an object of the Grid Search class
            grid = GridSearchCV(XGBRegressor(objective='reg:squarederror'), param_grid_xgboost, verbose=3,
                                cv=5)
            # finding the best parameters
            grid.fit(train_x, train_y)

            # extracting the best parameters
            learning_rate = grid.best_params_['learning_rate']
            max_depth = grid.best_params_['max_depth']
            n_estimators = grid.best_params_['n_estimators']

            # creating a new model with the best parameters objective='reg:linear'
            xgb = XGBRegressor(objective='reg:squarederror', learning_rate=learning_rate,
                               max_depth=max_depth,
                               n_estimators=n_estimators)
            # training the mew model
            xgb.fit(train_x, train_y)
            self.logger_object.log('XGBoost best params: ' + str(
                grid.best_params_) + '. Exited the get_best_params_for_xgboost method of the Model_Finder class')
            return xgb
        except Exception as e:
            model_finder = ModelFinderException(
                "Failed in [{0}] class [{1}] method [{2}]"
                    .format(self.__module__, ModelFinder.__name__,
                            self.get_best_params_for_xgboost_regressor.__name__))
            raise Exception(model_finder.error_message_detail(str(e), sys)) from e

    def get_best_model_zomato_or_fitbit_or_climate_visibility(self, train_x, train_y, test_x, test_y, cluster_no=None):
        """
        Method Name: get_best_model
        Description: Find out the Model which has the best AUC score.
        Output: The best model name and the model object
        On Failure: Raise Exception

        Written By: iNeuron Intelligence
        Version: 1.0
        Revisions: None

        """
        # create best model for KNN
        try:

            title = "Cluster {} ".format(cluster_no) if cluster_no is not None else ''
            self.model_name.append('Decision_Tree')

            self.logger_object.log('Entered the get_best_model method of the Model_Finder class')
            decision_tree_reg = self.get_best_params_for_decision_tree_regressor(train_x, train_y)

            self.model.append(decision_tree_reg)
            prediction_decision_tree_reg = decision_tree_reg.predict(
                test_x)  # Predictions using the decisionTreeReg Model
            decision_tree_reg_error = r2_score(test_y, prediction_decision_tree_reg)

            self.score.append(decision_tree_reg_error)
            self.logger_object.log("Decision tree regression r2 score {}".format(decision_tree_reg_error))

            # create best model for XGBoost
            self.model_name.append('XG_BOOST')
            xgboost = self.get_best_params_for_xgboost_regressor(train_x, train_y)
            prediction_xgboost = xgboost.predict(test_x)  # Predictions using the XGBoost Model

            self.model.append(xgboost)
            prediction_xgboost_error = r2_score(test_y, prediction_xgboost)
            self.logger_object.log("XGBoost regression r2 score {}".format(prediction_xgboost_error))
            self.score.append(prediction_xgboost_error)

            self.model_name.append('RIDGE_REG')
            ridge_regression = self.get_best_params_for_ridge_regression(train_x, train_y)
            self.model.append(ridge_regression)
            prediction_ridge_regression = ridge_regression.predict(test_x)
            prediction_ridge_error = r2_score(test_y, prediction_ridge_regression)
            self.score.append(prediction_ridge_error)
            self.logger_object.log("RIDGE_REG regression r2 score {}".format(prediction_ridge_error))

            self.model_name.append("Random_Forest")
            random_forest_reg = self.get_best_params_for_Random_Forest_Regressor(train_x, train_y)
            self.model.append(random_forest_reg)
            prediction_random_forest_reg = random_forest_reg.predict(test_x)
            prediction_random_forest_error = r2_score(test_y, prediction_random_forest_reg)
            self.score.append(prediction_random_forest_error)
            self.logger_object.log("Random Forest regression r2 score {}".format(prediction_ridge_error))

            self.model_name.append("SVR")
            sv_reg = self.get_best_params_for_support_vector_regressor(train_x, train_y)
            self.model.append(sv_reg)
            prediction_sv_reg = sv_reg.predict(test_x)
            prediction_sv_reg_error = r2_score(test_y, prediction_sv_reg)
            self.score.append(prediction_sv_reg_error)
            self.logger_object.log("Support vector regression r2 score {}".format(prediction_ridge_error))

            """
            Visualization begin based on above model
            """
            prediction_value = [prediction_decision_tree_reg,
                                prediction_xgboost,
                                prediction_ridge_regression,
                                prediction_random_forest_reg,
                                prediction_sv_reg]

            for data in zip(self.model_name, prediction_value):

                AccurayGraph().save_scatter_plot(x_axis_data=test_y, y_axis_data=data[1],
                                                 project_id=self.project_id,
                                                 execution_id=self.logger_object.execution_id,
                                                 file_object=self.file_object,
                                                 x_label="True Target values", y_label="Predicted Target value",
                                                 title=title + "Predicted vs True " + data[0])

                AccurayGraph().save_distribution_plot(data=numpy.abs(test_y - data[1]),
                                                      label="Residual distribution plot",
                                                      project_id=self.project_id,
                                                      execution_id=self.logger_object.execution_id,
                                                      file_object=self.file_object,
                                                      x_label="Error ",
                                                      y_label="frequency or occurrence",
                                                      title=title + "{} residual distribution plot".format(data[0])
                                                      )

            mean_abs_error = []
            for data in prediction_value:
                mean_abs_error.append(numpy.mean(numpy.abs(test_y - data)))

            AccurayGraph().save_accuracy_bar_graph(
                model_name_list=self.model_name,
                accuracy_score_list=mean_abs_error,
                project_id=self.project_id,
                execution_id=self.logger_object.execution_id,
                file_object=self.file_object,
                x_label="Model List",
                y_label="MAE comparison between {}".format(self.model_name),
                title=title + "Mean Absolute error "
            )
            execution_model_comparison_id = str(uuid.uuid4())
            for data in zip(self.model_name, self.score):
                self.save_accuracy_data(model_name=data[0], score=data[1],
                                        execution_model_comparision_id=execution_model_comparison_id)

            return self.get_best_model_on_score(model_name=self.model_name, model=self.model, score=self.score)
        except Exception as e:
            model_finder = ModelFinderException(
                "Failed in [{0}] class [{1}] method [{2}]"
                    .format(self.__module__, ModelFinder.__name__,
                            self.get_best_model_zomato_or_fitbit_or_climate_visibility.__name__))
            raise Exception(model_finder.error_message_detail(str(e), sys)) from e

    def get_best_params_for_naive_bayes(self, train_x, train_y):
        """
        Method Name: get_best_params_for_naive_bayes
        Description: get the parameters for the Naive Bayes's Algorithm which give the best accuracy.
                     Use Hyper Parameter Tuning.
        Output: The model with the best parameters
        On Failure: Raise Exception

        Written By: iNeuron Intelligence
        Version: 1.0
        Revisions: None

        """
        try:
            self.logger_object.log('Entered the get_best_params_for_naive_bayes method of the Model_Finder class')

            # initializing with different combination of parameters
            param_grid = {"var_smoothing": [1e-9, 0.1, 0.001, 0.5, 0.05, 0.01, 1e-8, 1e-7, 1e-6, 1e-10, 1e-11]}

            # Creating an object of the Grid Search class
            grid = GridSearchCV(estimator=self.gnb, param_grid=param_grid, cv=5, verbose=3)
            # finding the best parameters
            grid.fit(train_x, train_y)

            # extracting the best parameters
            var_smoothing = grid.best_params_['var_smoothing']

            # creating a new model with the best parameters
            gnb = GaussianNB(var_smoothing=var_smoothing)
            # training the mew model
            gnb.fit(train_x, train_y)
            self.logger_object.log('Naive Bayes best params: ' + str(
                grid.best_params_) + '. Exited the get_best_params_for_naive_bayes method of the Model_Finder class')

            return gnb
        except Exception as e:
            model_finder = ModelFinderException(
                "Failed in [{0}] class [{1}] method [{2}]".format(self.__module__, ModelFinder.__name__,
                                                                  self.get_best_params_for_naive_bayes.__name__))
            raise Exception(model_finder.error_message_detail(str(e), sys)) from e

    def get_best_params_for_xgboost_income_prediction(self, train_x, train_y):
        """
        Method Name: get_best_params_for_xgboost
        Description: get the parameters for XGBoost Algorithm which give the best accuracy.
        Use Hyper Parameter Tuning.
        Output: The model with the best parameters
        On Failure: Raise Exception

        Written By: iNeuron Intelligence
        Version: 1.0
        Revisions: None

        """
        try:
            self.logger_object.log('Entered the get_best_params_for_xgboost method of the Model_Finder class')
            # initializing with different combination of parameters
            param_grid_xgboost = {

                "n_estimators": [100, 130], "criterion": ['gini', 'entropy'],
                "max_depth": range(8, 10, 1)

            }
            # Creating an object of the Grid Search class
            grid = GridSearchCV(XGBClassifier(objective='binary:logistic'), param_grid_xgboost, verbose=3,
                                cv=5)
            # finding the best parameters
            grid.fit(train_x, train_y)

            # extracting the best parameters
            criterion = grid.best_params_['criterion']
            max_depth = grid.best_params_['max_depth']
            n_estimators = grid.best_params_['n_estimators']

            # creating a new model with the best parameters
            xgb = XGBClassifier(criterion=criterion, max_depth=max_depth, n_estimators=n_estimators,
                                n_jobs=-1)
            # training the mew model
            xgb.fit(train_x, train_y)
            self.logger_object.log('XGBoost best params: ' + str(
                grid.best_params_) + '. Exited the get_best_params_for_xgboost method of the Model_Finder class')
            return xgb
        except Exception as e:
            model_finder = ModelFinderException("Failed in [{0}] class [{1}] method [{2}]"
                                                .format(self.__module__,
                                                        ModelFinder.__name__,
                                                        self.get_best_params_for_xgboost_income_prediction.__name__))
            raise Exception(model_finder.error_message_detail(str(e), sys)) from e

    def get_best_model_income_prediction(self, train_x, train_y, test_x, test_y, cluster_number):
        """
                                                Method Name: get_best_model
                                                Description: Find out the Model which has the best AUC score.
                                                Output: The best model name and the model object
                                                On Failure: Raise Exception

                                                Written By: iNeuron Intelligence
                                                Version: 1.0
                                                Revisions: None

                                        """
        # create best model for XGBoost
        try:
            title_generator = " Cluster " + cluster_number + " model {}"

            # XG Boost model

            self.model_name.append('XG_BOOST')
            title = title_generator.format('XG_BOOST')
            self.logger_object.log('Entered the get_best_model method of the Model_Finder class')
            xgboost = self.get_best_params_for_xgboost_income_prediction(train_x, train_y)
            prediction_xgboost = xgboost.predict(test_x)  # Predictions using the XGBoost Model

            if len(test_y.unique()) == 1:  # if there is only one label in y, then roc_auc_score returns error. We
                # will use accuracy in that case
                xgboost_score = accuracy_score(test_y, prediction_xgboost)
                self.logger_object.log('Accuracy for XGBoost:' + str(xgboost_score))  # Log AUC
            else:
                xgboost_score = roc_auc_score(test_y, prediction_xgboost)  # AUC for XGBoost
                self.logger_object.log('AUC for XGBoost:' + str(xgboost_score))  # Log AUC
                y_score = xgboost.predict_proba(test_x)[:, 1]
                fpr, tpr, thresholds = roc_curve(test_y, y_score, pos_label=test_y.unique()[1])
                AccurayGraph().save_roc_curve_plot_binary_classification(fpr, tpr, self.project_id,
                                                                         execution_id=self.logger_object.execution_id,
                                                                         file_object=self.file_object,
                                                                         title=title)

            self.model.append(xgboost)
            self.score.append(xgboost_score)

            # create best model for naive bayes
            self.model_name.append('NAIVE_BAYES')
            title = title_generator.format('NAIVE_BAYES')
            naive_bayes = self.get_best_params_for_naive_bayes(train_x, train_y)
            prediction_naive_bayes = naive_bayes.predict(test_x)  # prediction using the Random Forest Algorithm
            self.model.append(naive_bayes)
            if len(test_y.unique()) == 1:  # if there is only one label in y,
                # then roc_auc_score returns error. We will use accuracy in that case
                naive_bayes_score = accuracy_score(test_y, prediction_naive_bayes)
                self.logger_object.log('Accuracy for naive bayes score' + str(naive_bayes_score))
            else:
                naive_bayes_score = roc_auc_score(test_y, prediction_naive_bayes)  # AUC for Random Forest
                self.logger_object.log('AUC for naive bayes score:' + str(naive_bayes_score))
                y_score = naive_bayes.predict_proba(test_x)[:, 1]
                fpr, tpr, thresholds = roc_curve(test_y, y_score, pos_label=test_y.unique()[0])
                AccurayGraph().save_roc_curve_plot_binary_classification(fpr, tpr, self.project_id,
                                                                         execution_id=self.logger_object.execution_id,
                                                                         file_object=self.file_object,
                                                                         title=title)

            self.score.append(naive_bayes_score)
            # create best model for Random forest
            self.model_name.append('Random_Forest')
            title = title_generator.format('Random_Forest')
            random_forest = self.get_best_params_for_random_forest(train_x, train_y)
            prediction_random_forest = random_forest.predict(test_x)
            self.model.append(random_forest)
            if len(test_y.unique()) == 1:
                random_forest_score = accuracy_score(test_y, prediction_random_forest)
                self.logger_object.log('Accuracy for Random Forest' + str(random_forest_score))
            else:
                random_forest_score = roc_auc_score(test_y, prediction_random_forest)  # AUC for Random Forest
                self.logger_object.log('AUC for Random Forest' + str(random_forest_score))
                y_score = random_forest.predict_proba(test_x)[:, 1]
                fpr, tpr, thresholds = roc_curve(test_y, y_score, pos_label=test_y.unique()[1])
                AccurayGraph().save_roc_curve_plot_binary_classification(fpr, tpr, self.project_id,
                                                                         execution_id=self.logger_object.execution_id,
                                                                         file_object=self.file_object,
                                                                         title=title)

            self.score.append(random_forest_score)

            # create best model for KNN
            self.model_name.append('KNN')
            title = title_generator.format('KNN')
            knn_clf = self.get_best_params_for_KNN(train_x, train_y)
            prediction_knn = knn_clf.predict(test_x)
            self.model.append(knn_clf)
            if len(test_y.unique()) == 1:
                knn_score = accuracy_score(test_y, prediction_knn)
                self.logger_object.log('Accuracy for KNN clf' + str(knn_score))
            else:
                knn_score = roc_auc_score(test_y, prediction_knn)  # AUC for Random Forest
                self.logger_object.log('AUC for KNN' + str(knn_score))
                y_score = knn_clf.predict_proba(test_x)[:, 1]
                fpr, tpr, thresholds = roc_curve(test_y, y_score, pos_label=test_y.unique()[1])
                AccurayGraph().save_roc_curve_plot_binary_classification(fpr, tpr, self.project_id,
                                                                         execution_id=self.logger_object.execution_id,
                                                                         file_object=self.file_object,
                                                                         title=title)
            self.score.append(knn_score)

            if len(test_y.unique()) != 1:
                """ 5. SVC """
                self.model_name.append("SVC")
                title = title_generator.format("SVC")
                svc_clf = self.get_best_params_for_svm_fraud_detection_and_scania(train_x, train_y)
                prediction_svc = svc_clf.predict(test_x)
                self.model.append(svc_clf)
                if len(test_y.unique()) == 1:
                    svc_score = accuracy_score(test_y, prediction_svc)
                    self.logger_object.log('Accuracy for svc clf' + str(svc_score))
                else:
                    svc_score = roc_auc_score(test_y, prediction_svc)  # AUC for Random Forest
                    self.logger_object.log('AUC for svc' + str(svc_score))
                    y_score = svc_clf.predict_proba(test_x)[:, 1]
                    fpr, tpr, thresholds = roc_curve(test_y, y_score, pos_label=test_y.unique()[1])
                    AccurayGraph().save_roc_curve_plot_binary_classification(fpr, tpr, self.project_id,
                                                                             execution_id=self.logger_object.execution_id,
                                                                             file_object=self.file_object,
                                                                             title=title)
                self.score.append(svc_score)

            AccurayGraph().save_accuracy_bar_graph(
                model_name_list=self.model_name,
                accuracy_score_list=self.score,
                project_id=self.project_id,
                execution_id=self.logger_object.execution_id,
                file_object=self.file_object,
                x_label="Model List",
                y_label="Accuracy score comparison {}".format(self.model_name),
                title="Cluster " + str(cluster_number) + "Accuracy Score "
            )

            execution_model_comparison_id = str(uuid.uuid4())
            for data in zip(self.model_name, self.score):
                self.save_accuracy_data(model_name=data[0], score=data[1],
                                        execution_model_comparision_id=execution_model_comparison_id)

            # comparing the two models
            return self.get_best_model_on_score(model_name=self.model_name, model=self.model, score=self.score)

        except Exception as e:
            model_finder = ModelFinderException(
                "Failed in [{0}] class [{1}] method [{2}]"
                    .format(self.__module__, ModelFinder.__name__,
                            self.get_best_model_income_prediction.__name__))
            raise Exception(model_finder.error_message_detail(str(e), sys)) from e

    def get_best_model_on_score(self, model_name: list, model: list, score: list):
        """

        :param model: models in list
        :param model_name: Model name list
        :param score: score list
        :return: best model name and model
        """
        try:
            record = {'model_name': model_name, 'model': model, 'score': score}
            df = pandas.DataFrame(record)
            df.index = df.model_name
            model_name = df.max()['model_name']
            model = df.loc[model_name]['model']
            return model_name, model

        except Exception as e:
            model_finder = ModelFinderException(
                "Failed in [{0}] class [{1}] method [{2}]".format(self.__module__, ModelFinder.__name__,
                                                                  self.get_best_model_on_score.__name__))
            raise Exception(model_finder.error_message_detail(str(e), sys)) from e
