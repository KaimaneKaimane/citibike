from code import store_model


def run():
    model_data = store_model.load_model_data()

    model = model_data['model']
    encoder = model_data['metadata']['encoder']

    prediction = model.predict([[1, 1, 1, 1, 1, 1, 1]])
    encoder.inverse_transform(prediction)


if __name__ == '__main__':
    run()
