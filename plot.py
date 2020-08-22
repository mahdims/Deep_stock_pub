import matplotlib.pyplot as plt
import numpy as np


def convert_the_date(date):
    date = str(date)
    Y = date[2:4]
    M = date[4:6]
    D = date[6:]
    return f"{Y}.{M}.{D}"


def date_of_tomorrow(today):
    today = today.split(".")
    today[2] = str(int(today[2])+1)
    tomorrow = '.'.join(today)
    return tomorrow


def plot_the_prediction(actual, forecast):
    prospective = len(forecast[0])
    forecast = np.asarray(forecast)
    dates = list(actual.date)
    dates.sort()
    dates = [convert_the_date(d) for d in dates]

    line, = plt.plot(dates, actual.closing_price, label="Real value",color='r', marker='o', markersize=5)
    color = ["b", "g", "w", "o"]
    lines = [line]
    for i in range(prospective):
        # add a new date at the end of predict
        tomorrow = date_of_tomorrow(dates[-1])
        dates.append(tomorrow)
        # delete the first date as we do not have prediction for that
        del dates[0]
        # Draw the plot
        line, = plt.plot(dates, forecast[:,i], color=color[i], label=f"{i+1} days a head", marker='+', markersize=5)
        lines.append(line)

    plt.legend(handles=lines, bbox_to_anchor=(0.96, 1), loc='upper left', borderaxespad=0.1)
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.title('Actual and predicted price')
    plt.show()
