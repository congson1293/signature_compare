import network
import joblib
import numpy as np
from keras.callbacks import EarlyStopping

def run():
    data = joblib.load('data/data.pkl')
    X, y = [], []
    for k, v in data.items():
        for vv in v:
            X.append(vv)
            y.append(k)

    model = network.building_network()
    print('Model summary...')
    print(model.summary())
    print('Training model...')

    early_stopping = EarlyStopping(patience=3)
    model.fit(np.array(X), np.array(y), batch_size=128, epochs=10, callbacks=[early_stopping])


if __name__ == '__main__':
    run()