
from code import load_data, preprocessing, store_model
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

extract_columns = [
    'tripduration',
    'starttime',
    'stoptime',
    'start station latitude',
    'start station longitude',
    'end station latitude',
    'end station longitude',
    'usertype',
    'birth year',
    'gender'
]


def prepare_data():
    print('Preprocessing data...')
    dataset = load_data.citibike(extract_columns)
    dataset = preprocessing.run(dataset, training=True)

    return preprocessing.encode_usertype(dataset)


def perform_train_test_split(dataset):
    print('Create train test split...')
    return train_test_split(
        dataset,
        test_size=0.33,
        random_state=42,
        stratify=dataset['usertype']
    )


def train_decision_tree(train_set):
    print('Train decision tree model...')
    decision_tree_model = DecisionTreeClassifier()
    decision_tree_model = decision_tree_model.fit(
        train_set.drop('usertype', axis=1),
        train_set['usertype']
    )
    return decision_tree_model


def evaluate_model(test_set, model):
    print('Evaluate decision tree model...')
    metadata = {}

    predicted_labels = model.predict(test_set.drop('usertype', axis=1))
    true_labels = test_set['usertype'].tolist()

    metadata['f1_score'] = f1_score(true_labels, predicted_labels)
    metadata['classification_report'] = classification_report(true_labels, predicted_labels)

    return metadata


def run():
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


