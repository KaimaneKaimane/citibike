import logging.config

from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from code.config import config, logger_config
from code import load_data, preprocessing, store_model


logging.config.dictConfig(logger_config)
logger = logging.getLogger(__name__)


def prepare_data():
    """
    Loads & preprocessed the citibike data.

    :return: the preprocessed citibike data
    """
    logger.info('Preprocessing data...')
    dataset = load_data.citibike(config['load']['extract_columns'])
    dataset = preprocessing.run(dataset, training=True)

    return preprocessing.encode_usertype(dataset)


def perform_train_test_split(dataset):
    """
    Creates a train test split.

    :param dataset: the citibike dataset
    :return: the train and test set
    """
    logger.info('Create train test split...')
    return train_test_split(
        dataset,
        stratify=dataset['usertype'],
        **config['training']['train_test_split']
    )


def train_decision_tree(train_set):
    """
    Trains a decision tree model.

    :param train_set: the citibike train set
    :return: the trained model
    """
    logger.info('Train decision tree model...')
    decision_tree_model = DecisionTreeClassifier()
    decision_tree_model = decision_tree_model.fit(
        train_set.drop(config['dataset']['label_column'], axis=1),
        train_set[config['dataset']['label_column']]
    )
    return decision_tree_model


def evaluate_model(test_set, model):
    """
    Evaluates the given model.

    :param test_set: the citibike test set
    :param model: the model to evaluate
    :return: dict containing the metrics
    """
    logger.info('Evaluate decision tree model...')
    metadata = {}

    predicted_labels = model.predict(test_set.drop(config['dataset']['label_column'], axis=1))
    true_labels = test_set[config['dataset']['label_column']].tolist()

    metadata['f1_score'] = f1_score(true_labels, predicted_labels)
    metadata['classification_report'] = classification_report(true_labels, predicted_labels)

    return metadata


def run():
    """
    Trains and saves the model.
    """
    citibike_dataset, usertype_encoder = prepare_data()
    citibike_train_set, citibike_test_set = perform_train_test_split(citibike_dataset)
    citibike_usertype_model = train_decision_tree(citibike_train_set)
    citibike_metadata = evaluate_model(citibike_test_set, citibike_usertype_model)
    citibike_metadata['encoder'] = usertype_encoder

    citibike_model_data = {
        'model': citibike_usertype_model,
        'metadata': citibike_metadata
    }

    store_model.save_model_data(citibike_model_data)


if __name__ == '__main__':
    run()


