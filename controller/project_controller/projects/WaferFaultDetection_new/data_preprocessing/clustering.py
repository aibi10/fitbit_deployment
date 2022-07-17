import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from kneed import KneeLocator
from controller.project_controller.projects.WaferFaultDetection_new.file_operations import file_methods
from project_library_layer.initializer.initializer import Initializer
import sys
from exception_layer.generic_exception.generic_exception import GenericException as KMeanClusteringModelException
from plotly_dash.accuracy_graph.accuracy_graph import AccurayGraph

class KMeansClustering:
    """
            This class shall  be used to divide the data into clusters before training.

            Written By: iNeuron Intelligence
            Version: 1.0
            Revisions: None

            """

    def __init__(self, project_id, file_object, logger_object):
        try:
            self.file_object = file_object
            self.logger_object = logger_object
            self.project_id = project_id
            self.initializer = Initializer()
        except Exception as e:
            model_exception = KMeanClusteringModelException(
                "Failed during object instantiation in module [{0}] class [{1}] method [{2}]"
                    .format(self.__module__, KMeansClustering.__name__,
                            self.__init__.__name__))
            raise Exception(model_exception.error_message_detail(str(e), sys)) from e

    def elbow_plot(self, data):
        """
                        Method Name: elbow_plot
                        Description: This method saves the plot to decide the optimum number of clusters to the file.
                        Output: A picture saved to the directory
                        On Failure: Raise Exception

                        Written By: iNeuron Intelligence
                        Version: 1.0
                        Revisions: None

                """
        # initializing an empty list
        try:
            self.logger_object.log('Entered the elbow_plot method of the KMeansClustering class')
            wcss = []
            for i in range(1, 11):
                kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)  # initializing the KMeans object
                kmeans.fit(data)  # fitting the data to the KMeans Algorithm
                wcss.append(kmeans.inertia_)
            plt.plot(range(1, 11), wcss)  # creating the graph between WCSS and the number of clusters
            plt.title('The Elbow Method')
            plt.xlabel('Number of clusters')
            plt.ylabel('WCSS')

            # plt.show()
            # plt.savefig('preprocessing_data/K-Means_Elbow.PNG') # saving the elbow plot locally
            # finding the value of the optimum cluster programmatically
            kn = KneeLocator(range(1, 11), wcss, curve='convex', direction='decreasing')
            AccurayGraph().save_line_plot(list(range(1, 11)),
                                          wcss, self.project_id,
                                          self.logger_object.execution_id,
                                          self.file_object,
                                          x_label='Number of clusters',
                                          y_label='WCSS',
                                          title="Optimum number of cluster "+str(kn.knee))
            self.logger_object.log(
                'The optimum number of clusters is: ' + str(kn.knee) + '. Exited the elbow_plot '
                                                                       'method of the '
                                                                       'KMeansClustering class')
            return kn.knee
        except Exception as e:
            model_exception = KMeanClusteringModelException(" Finding the number of clusters failed. Exited the "
                                                            "elbow_plot method of the KMeansClustering class in module "
                                                            "[{0}] class [{1}] method [{2}]"
                                                            .format(self.__module__, KMeansClustering.__name__,
                                                                    self.elbow_plot.__name__))
            raise Exception(model_exception.error_message_detail(str(e), sys)) from e

    def create_clusters(self, data, number_of_clusters):
        """
                                Method Name: create_clusters
                                Description: Create a new dataframe consisting of the cluster information.
                                Output: A datframe with cluster column
                                On Failure: Raise Exception

                                Written By: iNeuron Intelligence
                                Version: 1.0
                                Revisions: None

                        """

        try:
            self.logger_object.log('Entered the create_clusters method of the KMeansClustering class')
            kmeans = KMeans(n_clusters=number_of_clusters, init='k-means++', random_state=42)
            # self.data = self.data[~self.data.isin([np.nan, np.inf, -np.inf]).any(1)]
            y_kmeans = kmeans.fit_predict(data)  # divide data into clusters

            file_op = file_methods.FileOperation(self.project_id, self.file_object, self.logger_object)
            kmean_folder_name = self.initializer.get_kmean_folder_name()
            result = file_op.save_model(kmeans, kmean_folder_name)  # saving the KMeans model to directory
            # passing 'Model' as the functions need three parameters
            if result != 'success':
                raise Exception("cluster failed to save")
            data['Cluster'] = y_kmeans  # create a new column in dataset for storing the cluster information
            self.logger_object.log('successfully created ' + str(number_of_clusters) +
                                   'clusters. Exited the create_clusters method of the KMeansClustering class')
            return data
        except Exception as e:
            model_exception = KMeanClusteringModelException(" Fitting the data to clusters failed. Exited the "
                                                            "create_clusters method of the "
                                                            "KMeansClustering class module [{0}] class [{1}] method [{2}]"
                                                            .format(self.__module__, KMeansClustering.__name__,
                                                                    self.create_clusters.__name__))
            raise Exception(model_exception.error_message_detail(str(e), sys)) from e
