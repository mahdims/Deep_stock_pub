import numpy
import talib
from dataset import DataSet


def make_label(close_prices,start,prospective):
    y = []

    for i in range(prospective):
        y.append(close_prices[start+i]/close_prices[start+i-1] -1)

    return y


class Features(object):
    def __init__(self, selector, prospective):
        self.selector = selector
        self.supported = {"ROCP", "OROCP", "HROCP", "LROCP", "MACD", "RSI", "VROCP", "BOLL", "MA", "VMA",
                          "PRICE_VOLUME", "CROSS_PRICE"}
        self.feature = []
        self.prospective = prospective
        self.recall_period = 2

    def shift(self, xs, n):
        if n > 0:
            return numpy.r_[numpy.full(n, numpy.nan), xs[:-n]]
        elif n == 0:
            return xs
        else:
            return numpy.r_[xs[-n:], numpy.full(-n, numpy.nan)]

    def moving_extract(self, window=30, date=None, open_prices=None,
                       close_prices=None, high_prices=None, low_prices=None,
                       volumes=None, N_predict=1, flatten=True):

        self.extract(open_prices=open_prices, close_prices=close_prices, high_prices=high_prices, low_prices=low_prices,
                     volumes=volumes)

        feature_arr = numpy.asarray(self.feature)
        p = 0
        rows = feature_arr.shape[0]
        print("feature dimension: %s" % rows)
        all_data = DataSet([], [], [])
        predict = DataSet([], [], [])

        while p + window <= feature_arr.shape[1]:
            # The last self.prospective days can not produce complete labels
            if feature_arr.shape[1] - (p + window) >= N_predict:
                x = feature_arr[:, p:p + window]
                # Label the closing price of the next day -days
                y = make_label(close_prices, p + window, self.prospective)
                d = list(date[p + window: p + window + self.prospective])

                if flatten:
                    x = x.flatten("F")
                all_data.features.append(numpy.nan_to_num(x))
                all_data.labels.append(y)
                all_data.date.append(d)

            else:
                x = feature_arr[:, p:p + window]
                if flatten:
                    x = x.flatten("F")
                predict.features.append(numpy.nan_to_num(x))
                predict.date.append(date[p + window - 1])
                predict.closing_price.append(close_prices[p + window - 1])
                predict.last_label.append(close_prices[p + window - 2])
            p += 1

        all_data._features = numpy.asarray(all_data.features)
        all_data._labels = numpy.asarray(all_data.labels)
        all_data._date = numpy.asarray(all_data.date)
        predict._features = numpy.asarray(predict.features)
        predict._date = numpy.asarray(predict.date)
        predict._last_label = numpy.asarray(predict.last_label)
        predict._closing_price = numpy.asarray(predict.closing_price)

        return all_data, predict

    def extract(self, open_prices=None, close_prices=None, high_prices=None, low_prices=None, volumes=None):
        self.feature = []
        for feature_type in self.selector:
            if feature_type in self.supported:
                print("extracting feature : %s" % feature_type)
                self.extract_by_type(feature_type, open_prices=open_prices, close_prices=close_prices,
                                     high_prices=high_prices, low_prices=low_prices, volumes=volumes)
            else:
                print("feature type not supported: %s" % feature_type)
        self.feature_distribution()
        return self.feature

    def feature_distribution(self):
        k = 0
        for feature_column in self.feature:
            fc = numpy.nan_to_num(feature_column)
            mean = numpy.mean(fc)
            var = numpy.var(fc)
            max_value = numpy.max(fc)
            min_value = numpy.min(fc)
            print("[%s_th feature] mean: %s, var: %s, max: %s, min: %s" % (k, mean, var, max_value, min_value))
            k = k + 1

    def extract_by_type(self, feature_type, open_prices=None, close_prices=None, high_prices=None, low_prices=None,
                        volumes=None):
        N_row = len(open_prices)
        periods = [5*a for a in [1,2,4,6,12,24,48, 90]]
        if feature_type == 'ROCP':
            for i in range(1, self.recall_period):
                self.feature.append(talib.ROCP(close_prices, timeperiod=i) / float(i))

        if feature_type == 'OROCP':
            for i in range(1, self.recall_period):
                self.feature.append(talib.ROCP(open_prices, timeperiod=i) / float(i))

        if feature_type == 'HROCP':
            for i in range(1, self.recall_period):
                self.feature.append(talib.ROCP(high_prices, timeperiod=i) / float(i))

        if feature_type == 'LROCP':
            for i in range(1, self.recall_period):
                self.feature.append(talib.ROCP(low_prices, timeperiod=i) / float(i))

        if feature_type == 'MACD':
            macd, signal, hist = talib.MACD(close_prices, fastperiod=12, slowperiod=26, signalperiod=9)
            norm_signal = numpy.minimum(numpy.maximum(numpy.nan_to_num(signal), -1), 1)
            norm_hist = numpy.minimum(numpy.maximum(numpy.nan_to_num(hist), -1), 1)
            norm_macd = numpy.minimum(numpy.maximum(numpy.nan_to_num(macd), -1), 1)

            zero = numpy.asarray([0])
            macdrocp = numpy.minimum(numpy.maximum(numpy.concatenate((zero, numpy.diff(numpy.nan_to_num(macd)))),-1),1)
            signalrocp = numpy.minimum(
                numpy.maximum(numpy.concatenate((zero, numpy.diff(numpy.nan_to_num(signal)))), -1), 1)
            histrocp = numpy.minimum(numpy.maximum(numpy.concatenate((zero, numpy.diff(numpy.nan_to_num(hist)))), -1),1)

            self.feature.append(norm_macd)
            self.feature.append(norm_signal)
            self.feature.append(norm_hist)

            self.feature.append(macdrocp)
            self.feature.append(signalrocp)
            self.feature.append(histrocp)

        if feature_type == 'RSI':
            for time_period in [6,12,24]:
                rsi = talib.RSI(close_prices, timeperiod=time_period)
                rsirocp = talib.ROCP(rsi + 100., timeperiod=1)
                self.feature.append(rsi / 100.0 - 0.5)
                self.feature.append(rsirocp)


            # self.feature.append(numpy.maximum(rsi6 / 100.0 - 0.8, 0))
            # self.feature.append(numpy.maximum(rsi12 / 100.0 - 0.8, 0))
            # self.feature.append(numpy.maximum(rsi24 / 100.0 - 0.8, 0))
            # self.feature.append(numpy.minimum(rsi6 / 100.0 - 0.2, 0))
            # self.feature.append(numpy.minimum(rsi6 / 100.0 - 0.2, 0))
            # self.feature.append(numpy.minimum(rsi6 / 100.0 - 0.2, 0))
            # self.feature.append(numpy.maximum(numpy.minimum(rsi6 / 100.0 - 0.5, 0.3), -0.3))
            # self.feature.append(numpy.maximum(numpy.minimum(rsi6 / 100.0 - 0.5, 0.3), -0.3))
            # self.feature.append(numpy.maximum(numpy.minimum(rsi6 / 100.0 - 0.5, 0.3), -0.3))

        if feature_type == 'VROCP':
            for i in range(1, self.recall_period):
                self.feature.append(numpy.arctan(numpy.nan_to_num(talib.ROCP(numpy.maximum(volumes, 1), timeperiod=i))))

        if feature_type == 'BOLL':
            upperband, middleband, lowerband = talib.BBANDS(close_prices, timeperiod=5, nbdevup=2, nbdevdn=2, matype=0)
            self.feature.append((upperband - close_prices) / close_prices)
            self.feature.append((middleband - close_prices) / close_prices)
            self.feature.append((lowerband - close_prices) / close_prices)

        if feature_type == 'MA':
            for n in periods:
                ma = numpy.nan_to_num(talib.MA(close_prices, timeperiod=n))
                marocp = talib.ROCP(ma, timeperiod=1)
                self.feature.append(marocp)
                self.feature.append((ma - close_prices) / close_prices)

        if feature_type == 'VMA':
            for n in periods:
                ma = numpy.nan_to_num(talib.MA(volumes, timeperiod=n))
                marocp = numpy.tanh(numpy.nan_to_num(talib.ROCP(ma, timeperiod=1)))
                self.feature.append(marocp)
                self.feature.append(numpy.tanh(numpy.nan_to_num((ma - volumes) / (volumes + 1))))

        if feature_type == 'PRICE_VOLUME':
            rocp = talib.ROCP(close_prices, timeperiod=1)
            # norm_volumes = (volumes - numpy.mean(volumes)) / math.sqrt(numpy.var(volumes))
            # vrocp = talib.ROCP(norm_volumes + numpy.max(norm_volumes) - numpy.min(norm_volumes), timeperiod=1)
            vrocp = numpy.tanh(numpy.nan_to_num(talib.ROCP(numpy.maximum(volumes, 1), timeperiod=1)))
            pv = rocp * vrocp
            self.feature.append(pv)

        if feature_type == 'CROSS_PRICE':
            for i in range(0, self.recall_period - 1):
                shift_open = self.shift(open_prices, i)
                shift_close = self.shift(close_prices, i)
                shift_high = self.shift(high_prices, i)
                shift_low = self.shift(low_prices, i)
                for list_inx, price in enumerate([close_prices, high_prices, low_prices, open_prices]):

                    if list_inx != 3: # price != open_prices:
                        self.feature.append(
                            numpy.minimum(numpy.maximum(numpy.nan_to_num((price - shift_open) / shift_open), -1), 1))

                    if list_inx != 0: # price != close_prices:
                        self.feature.append(
                            numpy.minimum(numpy.maximum(numpy.nan_to_num((price - shift_close) / shift_close), -1), 1))

                    if list_inx != 1: # price != high_prices:
                        self.feature.append(
                            numpy.minimum(numpy.maximum(numpy.nan_to_num((price - shift_high) / shift_high), -1), 1))

                    if list_inx != 2: # price != low_prices:
                        self.feature.append(
                            numpy.minimum(numpy.maximum(numpy.nan_to_num((price - shift_low) / shift_low), -1), 1))


def extract_feature(raw_data, selector, prospective, window=30, N_predict=1, flatten=True):
    chart_feature = Features(selector, prospective)
    sorted_data = sorted(raw_data, key=lambda x: x.date)
    closes = []
    opens = []
    highs = []
    lows = []
    volumes = []
    date = []
    for item in sorted_data:
        date.append(item.date)
        closes.append(item.close)
        opens.append(item.open)
        highs.append(item.high)
        lows.append(item.low)
        volumes.append(float(item.volume))
    date = numpy.asarray(date)
    closes = numpy.asarray(closes)
    opens = numpy.asarray(opens)
    highs = numpy.asarray(highs)
    lows = numpy.asarray(lows)
    volumes = numpy.asarray(volumes)

    all_data, predict = chart_feature.moving_extract(window=window, date=date,
                                                                               open_prices=opens,
                                                                               close_prices=closes,
                                                                               high_prices=highs, low_prices=lows,
                                                                               volumes=volumes,
                                                                               N_predict=N_predict,
                                                                               flatten=flatten)

    all_data.label_normalization()

    return all_data, predict

