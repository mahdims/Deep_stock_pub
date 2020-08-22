from math import ceil
from rawdata import RawData, read_sample_data
from dataset import DataSet
from chart import extract_feature
import pickle

# This script will extract the features out of the raw data, create the train and test data sets. And save them for
# later use.


def extract_with_filename(filepath, selector, test_percentage = 0.2,prospective = 1 ,window=30, N_predict=3):

    raw_data = read_sample_data(filepath)
    All_train, predict_data = extract_feature(raw_data=raw_data,
                                                      selector=selector,
                                                      prospective=prospective,
                                                      window=window,
                                                      N_predict =N_predict,
                                                      flatten=False)


    print("feature extraction done, start writing to file...")

    train_data, test_data = Train_test(All_train, test_percentage)

    return train_data, test_data, predict_data


def Train_test(all_data, test_percentage):
    # This function will divide the test and train data sets
    # Note that the test data set is used as validation and predict data used as the pure test
    N_rows = all_data.labels.shape[0]
    N_test = int(N_rows * test_percentage)
    N_months = ceil(N_rows/30) # Number of periods
    step = int(N_test / N_months)  # number of test samples from each period
    Train_inx = []
    Test_inx = []
    for i in range(0, N_rows-30, 30):
        Train_inx.extend(range(i, i+30-step))
        Test_inx.extend(range(i+30-step, i+30))
    # As the last period is less than one full month we should select the test equal to step size and leave the rest
    # for the train
    if N_rows - (i+30) > step:
        Train_inx.extend(range(i+30, N_rows-step))
        Test_inx.extend(range(N_rows-step, N_rows))
    else:
        Test_inx.extend(range(i+30, N_rows))

    Train = DataSet(all_data.features[Train_inx], all_data.labels[Train_inx], all_data.date[Train_inx])
    Train.label_max = all_data.label_max
    Train.label_min = all_data.label_min
    Test = DataSet(all_data.features[Test_inx], all_data.labels[Test_inx], all_data.date[Test_inx])

    return Train, Test


def save_in_file(path2file, train, test, predict):

    train_set = {"code": filename[:filename.rfind(".")], "feature": train.features, "label": train.labels,
                 "date": train.date, "MAX": train.label_max, "MIN": train.label_min}
    pickle.dump(train_set, fp, 2)

    test_set = {"code": filename[:filename.rfind(".")], "feature": test.features, "label": test.labels, "date": test.date}
    pickle.dump(test_set, fp, 2)

    predict_set = {"code": filename[:filename.rfind(".")], "feature": predict.features, "last_label": predict.last_label,
                   "date": predict.date, "closing_price": predict.closing_price}

    pickle.dump(predict_set, fp, 2)


if __name__ == '__main__':
    test_percentage = 0.2
    prospective = 1
    N_predict = 10
    assert N_predict > prospective
    # input_shape = [30, 61]  # [length of time series, length of feature]
    window = 20  # input_shape[0]

    selector = ["ROCP", "OROCP", "HROCP", "LROCP", "MACD", "RSI", "VROCP",
                "BOLL", "MA", "VMA", "PRICE_VOLUME", "CROSS_PRICE"]

    dataset_dir = "./dataset"

    # We save all the train-test-predict data sets for all input files in on pickle.
    # So the open should be out of function. But the
    fp = open("ultimate_feature", "wb")
    # for filename in os.listdir(dataset_dir): # all data files in data set path
    for filename in ["HiWeb.csv"]:
        print("processing file: " + filename)
        filepath = dataset_dir + "/" + filename

        train_data, test_data, predict_data = extract_with_filename(filepath, selector, test_percentage=test_percentage,
                                                                    prospective=prospective, window=window,
                                                                    N_predict=N_predict)
        save_in_file(filepath, train_data, test_data, predict_data)

    fp.close()
