import sys
import os
import numpy
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
from Model import Model, risk_estimation, pairwise_logit
from dataset import DataSet
import pickle
from plot import plot_the_prediction


def read_feature(path):
    train_features = []
    train_labels = []
    train_date =[]
    test_features = []
    test_labels = []
    test_date =[]
    with open(path, "rb") as fp:
        while True:
            try:
                train_map = pickle.load(fp)
                test_map = pickle.load(fp)
                predict_map = pickle.load(fp)

                train_features.extend(train_map["feature"])
                train_labels.extend(train_map["label"])
                train_date.extend(train_map["date"])
                train_set = DataSet(numpy.transpose(numpy.asarray(train_features), [0, 2, 1]), numpy.asarray(train_labels),
                        numpy.asarray(train_date))

                test_features.extend(test_map["feature"])
                test_labels.extend(test_map["label"])
                test_date.extend(test_map["date"])
                test_set = DataSet(numpy.transpose(numpy.asarray(test_features), [0, 2, 1]), numpy.asarray(test_labels),
                                  numpy.asarray(test_date))

                predict_feature = predict_map["feature"]
                predict_date = predict_map["date"]
                predict_set = DataSet(numpy.transpose(numpy.asarray(predict_feature), [0, 2, 1]), numpy.asarray(predict_date),
                        numpy.asarray(predict_date))
                predict_set._last_label = predict_map["last_label"]
                predict_set._closing_price = predict_map["closing_price"]

                Max_label = train_map["MAX"]
                Min_label = train_map["MIN"]

                print("read %s successfully. " % train_map["code"])

            except Exception as e:
                break
    return train_set, test_set, predict_set, Max_label, Min_label


def read_separate_feature(path):
    train_sets = {}
    test_sets = {}
    with open(path, "rb") as fp:
        while True:
            try:
                train_map = pickle.load(fp)
                test_map = pickle.load(fp)
                train_sets[train_map["code"]] = DataSet(numpy.transpose(numpy.asarray(train_map["feature"]), [0, 2, 1]),
                                                        numpy.asarray(train_map["label"]))
                test_sets[test_map["code"]] = DataSet(numpy.transpose(numpy.asarray(test_map["feature"]), [0, 2, 1]),
                                                      numpy.asarray(test_map["label"]))
                print("read %s successfully. " % train_map["code"])
            except Exception as e:
                break

    return train_sets, test_sets


def make_model(nb_epochs=100, batch_size=128, lr=0.01, n_layers=1, n_hidden=14, rate_dropout=0.3, loss=risk_estimation):
    # Number of days to predict a head
    prospective = 2
    # read the train an set data based
    train_set, test_set, _, _, _ = read_feature("./ultimate_feature")  # read_ultimate("./", input_shape)
    input_shape = [train_set.features.shape[1], train_set.features.shape[2]]
    model_path = './checkpoint/model.%s' % input_shape[0] + '.best'
    wp = Model(input_shape=input_shape, lr=lr, n_layers=n_layers, n_hidden=n_hidden, rate_dropout=rate_dropout,
                    loss=loss)
    wp.build_model(prospective)

    # Why validation is the test date !!!!!!!
    wp.fit(train_set.features, train_set.labels, batch_size=batch_size,
           nb_epoch=nb_epochs, shuffle=True, verbose=1,
           validation_data=(test_set.features, test_set.labels),
           callbacks=[TensorBoard(histogram_freq=10),
                      ModelCheckpoint(filepath=model_path, save_best_only=True, mode='min')])
    scores = wp.evaluate(test_set.features, test_set.labels, verbose=0)
    print('Test loss:', scores[0])
    print('Test accuracy:', scores[1])

    wp.model.save(model_path)
    saved_wp = wp.load_model(model_path)
    scores = saved_wp.evaluate(test_set.features, test_set.labels, verbose=0)
    print('Test loss:', scores[0])
    print('test accuracy:', scores[1])
    pred = saved_wp.predict(test_set.features, 1024)
    # print(pred)
    # print(test_set.labels)
    pred = numpy.reshape(pred, [-1])
    result = numpy.array([pred, test_set.labels]).transpose()
    with open('output.' + str(input_shape[0]), 'w') as fp:
        for i in range(result.shape[0]):
            for val in result[i]:
                fp.write(str(val) + "\t")
            fp.write('\n')


def make_separate_model(nb_epochs=100, batch_size=128, lr=0.01, n_layers=1, n_hidden=14, rate_dropout=0.3, input_shape=[30, 73]):
    train_sets, test_sets = read_separate_feature("./ultimate_feature")

    wp = WindPuller(input_shape=input_shape, lr=lr, n_layers=n_layers, n_hidden=n_hidden, rate_dropout=rate_dropout)
    wp.build_model()
    for code, train_set in train_sets.items():
        test_set = test_sets[code]
        input_shape = [train_set.features.shape[1], train_set.features.shape[2]]
        print(input_shape)
        model_path = 'model.%s' % code

        print(train_set.features.shape)
        wp.fit(train_set.features, train_set.labels, batch_size=batch_size,
               nb_epoch=nb_epochs, shuffle=False, verbose=1,
               validation_data=(test_set.features, test_set.labels),
               callbacks=[TensorBoard(histogram_freq=1000),
                          ModelCheckpoint(filepath=model_path + '.best', save_best_only=True, mode='min')])
        scores = wp.evaluate(test_set.features, test_set.labels, verbose=0)
        print('Test loss:', scores[0])
        print('Test accuracy:', scores[1])

        wp.model.save(model_path)
        saved_wp = wp.load_model(model_path)
        scores = saved_wp.evaluate(test_set.features, test_set.labels, verbose=0)
        print('Test loss:', scores[0])
        print('test accuracy:', scores[1])
        pred = saved_wp.predict(test_set.features, 1024)
        # print(pred)
        # print(test_set.labels)
        pred = numpy.reshape(pred, [-1])
        result = numpy.array([pred, test_set.labels]).transpose()
        with open('output.' + str(input_shape[0]), 'w') as fp:
            for i in range(result.shape[0]):
                for val in result[i]:
                    fp.write(str(val) + "\t")
                fp.write('\n')


def evaluate_model(model_path, code, input_shape=[30, 61]):

    _, _, predict_set, Max, Min = read_feature("./%s_feature" % code)
    saved_wp = Model.load_model(model_path)

    raw_pred = saved_wp.predict(predict_set.features)
    Price_prediction = []

    print("Date \t Forecast")
    for i in range(len(raw_pred)):
        # print(pred[i])
        last_real_price = predict_set.last_label[i]
        price = [round(last_real_price*(1+a*(Max-Min) + Min), 4) for a in raw_pred[i]]
        Price_prediction.append(price)
        print(f"{predict_set.date[i]}\t {price}")
        # print(f"{[round(a,4) for a in test_set.labels[i]]} \t {[round(a,4) for a in pred[i]]}")
    plot_the_prediction(predict_set, Price_prediction)


if __name__ == '__main__':
    #operation = "predict"
    operation = "train"
    # input_shape = [30, 102]
    if len(sys.argv) > 1:
        operation = sys.argv[1]
    if operation == "train":
        # make_separate_model(10000, 512, lr=0.0005, n_hidden=14, rate_dropout=0.5, input_shape=[30, 73])
        make_model(nb_epochs=1000, batch_size=50, lr=0.01, n_hidden=14, rate_dropout=0.4)
        # make_model(30000, 512, lr=0.01, n_hidden=64, rate_dropout=0.5, loss=pairwise_logit)
    elif operation == "predict":
        CWD = os.getcwd()
        path2model = CWD + "/checkpoint/model.20.best"
        evaluate_model(path2model, "ultimate")
