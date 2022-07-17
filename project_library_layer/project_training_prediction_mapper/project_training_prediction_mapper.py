from controller.project_controller.projects.fraud_detection.training_model_fraud import TrainingModel as TrainModelFraud
from controller.project_controller.projects.fraud_detection.prediction_model_fraud import Prediction as PredictionFraud
from controller.project_controller.projects.WaferFaultDetection_new.training_Validation_Insertion import \
    TrainingValidation
from controller.project_controller.projects.WaferFaultDetection_new.prediction_Validation_Insertion import \
    PredictionValidation
from controller.project_controller.projects.WaferFaultDetection_new.trainingModel import \
    TrainingModel as TrainModelWafer
from controller.project_controller.projects.WaferFaultDetection_new.predictFromModel import \
    Prediction as PredictionWafer
from controller.project_controller.projects.WaferFaultDetection_new.trainingModel import \
    TrainingModel as TrainModelThyroid
from controller.project_controller.projects.WaferFaultDetection_new.predictFromModel import \
    Prediction as PredictionThyroid

from controller.project_controller.projects.mushroom.train_model_murshroom import TrainingModel as TrainModelMushroom
from controller.project_controller.projects.mushroom.predict_from_model_mushroom import Prediction as PredictionMushroom
from controller.project_controller.projects.fraud_detection.traning_validation_insertion import TrainingValidation as \
    TrainingValidationFraudDetection
from controller.project_controller.projects.fraud_detection.prediction_validation_insertion import PredictionValidation \
    as PredictionValidationFraudDetection

from controller.project_controller.projects.credit_card_default.prediction_model_credit_defaulter import Prediction \
    as PredictCreditDefaulters
from controller.project_controller.projects.credit_card_default.prediction_validation_insertion import  PredictionValidation \
    as PredictionValidationCreditDefaulters
from controller.project_controller.projects.credit_card_default.training_model_credit_deaulter import TrainingModel \
    as TrainCreditDefaulters
from controller.project_controller.projects.credit_card_default.training_validation_insertion import TrainingValidation \
    as TrainingValidationCreditDefaulters
"""Phising classifier """

from controller.project_controller.projects.phising_classifier.prediction_model_phising_classifier import Prediction \
    as PredictPhisingClassifier
from controller.project_controller.projects.phising_classifier.prediction_validation_insertion import  PredictionValidation \
    as PredictionValidationPhisingClassifier
from controller.project_controller.projects.phising_classifier.training_model_phising_classifier import TrainingModel \
    as TrainPhisingClassifier
from controller.project_controller.projects.phising_classifier.training_validation_insertion import TrainingValidation \
    as TrainingValidationPhisingClassifier

"""Forest classifier"""

from controller.project_controller.projects.forest_cover_classification.prediction_model_forset_cover import Prediction \
    as PredictForesetCoverClassifier
from controller.project_controller.projects.forest_cover_classification.prediction_validation_insertion import  PredictionValidation \
    as PredictionValidationForesetCoverClassifier
from controller.project_controller.projects.forest_cover_classification.training_model_forest_cover import TrainingModel \
    as TrainForesetCoverClassifier
from controller.project_controller.projects.forest_cover_classification.training_validation_insertion import TrainingValidation \
    as TrainingValidationForesetCoverClassifier
"""Scania AB truck """
from controller.project_controller.projects.scania_truck.prediction_model_scania_truck import Prediction \
    as PredictScaniaTruckPressure
from controller.project_controller.projects.scania_truck.prediction_validation_insertion import  PredictionValidation \
    as PredictionValidationScaniaTruckPressure
from controller.project_controller.projects.scania_truck.training_model_scania_truck import TrainingModel \
    as TrainScaniaTruckPressure
from controller.project_controller.projects.scania_truck.traning_validation_insertion import TrainingValidation \
    as TrainingValidationScaniaTruckPressure
"""Back Order"""
from controller.project_controller.projects.back_order.prediction_model_back_order import Prediction \
    as PredictBackOrder
from controller.project_controller.projects.back_order.prediction_validation_insertion import  PredictionValidation \
    as PredictionValidationBackOrder
from controller.project_controller.projects.back_order.training_model_back_order import TrainingModel \
    as TrainBackOrder
from controller.project_controller.projects.back_order.training_validation_insertion import TrainingValidation \
    as TrainingValidationBackOrder
"""BigMartSales"""
from controller.project_controller.projects.bigmart_sales.prediction_model_bigmart_sales import Prediction \
    as PredictBigMartSales
from controller.project_controller.projects.bigmart_sales.prediction_validation_insertion import  PredictionValidation \
    as PredictionValidationBigMartSales
from controller.project_controller.projects.bigmart_sales.training_model_bigmart_sales import TrainingModel \
    as TrainBigMartSales
from controller.project_controller.projects.bigmart_sales.training_validation_insertion import TrainingValidation \
    as TrainingValidationBigMartSales
"""cement strength """
from controller.project_controller.projects.cement_strength.prediction_model_cement_strength import Prediction \
    as PredictCementStrength
from controller.project_controller.projects.cement_strength.prediction_validation_insertion import  PredictionValidation \
    as PredictionValidationCementStrength
from controller.project_controller.projects.cement_strength.training_model_cement_strength import TrainingModel \
    as TrainCementStrength
from controller.project_controller.projects.cement_strength.training_validation_insertion import TrainingValidation \
    as TrainingValidationCementStrength

"""zomato"""
from controller.project_controller.projects.zomato.prediction_model_zomato import Prediction \
    as PredictZomato
from controller.project_controller.projects.zomato.prediction_validation_insertion import  PredictionValidation \
    as PredictionValidationZomato
from controller.project_controller.projects.zomato.train_model_zomato import TrainingModel \
    as TrainZomato
from controller.project_controller.projects.zomato.training_validation_insertion import TrainingValidation \
    as TrainingValidationZomato

"""fitbit"""
from controller.project_controller.projects.fitbit.prediction_model_fitbit import Prediction \
    as PredictFitBit
from controller.project_controller.projects.fitbit.prediction_validation_insertion import  PredictionValidation \
    as PredictionValidationFitBit
from controller.project_controller.projects.fitbit.train_model_fitbit import TrainingModel \
    as TrainFitBit
from controller.project_controller.projects.fitbit.training_validation_insertion import TrainingValidation \
    as TrainingValidationFitBit

"""Climate Visibility"""
from controller.project_controller.projects.visibility_climate.prediction_model_visibility_climate import Prediction \
    as PredictVisibilityClimate
from controller.project_controller.projects.visibility_climate.prediction_validation_insertion import  PredictionValidation \
    as PredictionValidationVisibilityClimate
from controller.project_controller.projects.visibility_climate.train_model_visibility_climate import TrainingModel \
    as TrainVisibilityClimate
from controller.project_controller.projects.visibility_climate.training_validation_insertion import TrainingValidation \
    as TrainingValidationVisibilityClimate


"""Income Prediction"""
from controller.project_controller.projects.income_prediction.prediction_model_income_prediction import Prediction \
    as PredictIncomePrediction
from controller.project_controller.projects.income_prediction.prediction_validation_insertion import  PredictionValidation \
    as PredictionValidationIncomePrediction
from controller.project_controller.projects.income_prediction.train_model_income_prediction import TrainingModel \
    as TrainIncomePrediction
from controller.project_controller.projects.income_prediction.training_validation_insertion import TrainingValidation \
    as TrainingValidationIncomePrediction

""" Sentiment Analysis"""
from controller.project_controller.projects.sentiment_analysis.sentiment_analysis_deploy.clientApp import ClientApi

project_train_and_prediction_mapper = [
    {
        'project_id': 1,
        'training_class_name': TrainModelWafer,
        'prediction_class_name': PredictionWafer,
        'training_validation_class_name': TrainingValidation,
        'prediction_validation_class_name': PredictionValidation
    },
    {
        'project_id': 2,
        'training_class_name': TrainModelThyroid,
        'prediction_class_name': PredictionThyroid,
        'training_validation_class_name': TrainingValidation,
        'prediction_validation_class_name': PredictionValidation
    },
    {
        'project_id': 3,
        'training_class_name': TrainModelMushroom,
        'prediction_class_name': PredictionMushroom,
        'training_validation_class_name': TrainingValidation,
        'prediction_validation_class_name': PredictionValidation
    },
    {
        'project_id': 4,
        'training_class_name': TrainModelFraud,
        'prediction_class_name': PredictionFraud,
        'training_validation_class_name': TrainingValidationFraudDetection,
        'prediction_validation_class_name': PredictionValidationFraudDetection
    },
    {
        'project_id': 5,
        'training_class_name': TrainCreditDefaulters,
        'prediction_class_name': PredictCreditDefaulters,
        'training_validation_class_name': TrainingValidationCreditDefaulters,
        'prediction_validation_class_name': PredictionValidationCreditDefaulters
    },
    {
        'project_id': 6,
        'training_class_name': TrainPhisingClassifier,
        'prediction_class_name': PredictPhisingClassifier,
        'training_validation_class_name': TrainingValidationPhisingClassifier,
        'prediction_validation_class_name': PredictionValidationPhisingClassifier
    },
    {
        'project_id': 7,
        'training_class_name': TrainForesetCoverClassifier,
        'prediction_class_name': PredictForesetCoverClassifier,
        'training_validation_class_name': TrainingValidationForesetCoverClassifier,
        'prediction_validation_class_name': PredictionValidationForesetCoverClassifier
    },
    {
        'project_id': 8,
        'training_class_name': TrainScaniaTruckPressure,
        'prediction_class_name': PredictScaniaTruckPressure,
        'training_validation_class_name': TrainingValidationScaniaTruckPressure,
        'prediction_validation_class_name': PredictionValidationScaniaTruckPressure
    },
    {
        'project_id': 9,
        'training_class_name': TrainBackOrder,
        'prediction_class_name': PredictBackOrder,
        'training_validation_class_name': TrainingValidationBackOrder,
        'prediction_validation_class_name': PredictionValidationBackOrder
    },
    {
        'project_id': 10,
        'training_class_name': TrainBigMartSales,
        'prediction_class_name': PredictBigMartSales,
        'training_validation_class_name': TrainingValidationBigMartSales,
        'prediction_validation_class_name': PredictionValidationBigMartSales
    },
    {
        'project_id': 11,
        'training_class_name': TrainCementStrength,
        'prediction_class_name': PredictCementStrength,
        'training_validation_class_name': TrainingValidationCementStrength,
        'prediction_validation_class_name': PredictionValidationCementStrength
    },
    {
        'project_id': 12,
        'training_class_name': TrainZomato,
        'prediction_class_name': PredictZomato,
        'training_validation_class_name': TrainingValidationZomato,
        'prediction_validation_class_name': PredictionValidationZomato
    },
    {
        'project_id': 13,
        'training_class_name': TrainFitBit,
        'prediction_class_name': PredictFitBit,
        'training_validation_class_name': TrainingValidationFitBit,
        'prediction_validation_class_name': PredictionValidationFitBit
    },
    {
        'project_id': 14,
        'training_class_name': TrainVisibilityClimate,
        'prediction_class_name': PredictVisibilityClimate,
        'training_validation_class_name': TrainingValidationVisibilityClimate,
        'prediction_validation_class_name': PredictionValidationVisibilityClimate
    },
    {
        'project_id': 15,
        'training_class_name': TrainIncomePrediction,
        'prediction_class_name': PredictIncomePrediction,
        'training_validation_class_name': TrainingValidationIncomePrediction,
        'prediction_validation_class_name': PredictionValidationIncomePrediction
    },
    {
        'project_id': 16,
        'training_class_name':ClientApi,
        'prediction_class_name':ClientApi,
        'training_validation_class_name': None,
        'prediction_validation_class_name': None
    },

]


def get_training_validation_and_training_model_class_name(project_id):
    try:
        for i in project_train_and_prediction_mapper:
            if i['project_id'] == project_id:
                return i['training_validation_class_name'], i['training_class_name'],

    except Exception as e:
        raise e


def get_prediction_validation_and_prediction_model_class_name(project_id):
    try:
        for i in project_train_and_prediction_mapper:
            if i['project_id'] == project_id:
                return i['prediction_validation_class_name'], i['prediction_class_name']
    except Exception as e:
        raise e
