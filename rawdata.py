
class RawData(object):
    def __init__(self, date, open, high, low, close, value, volume):
        self.date = date
        self.open = open
        self.high = high
        self.close = close
        self.low = low
        self.volume = volume
        self.value = value


def read_sample_data(path):
    print("reading histories...")
    raw_data = []
    separator = ","
    with open(path, "r") as fp:
        for line in fp:
            if line.startswith("<TICKER>"):  # ignore label line
                continue
            line = line.split(separator)[1:8]
            fields = line
            # if len(fields) >= 8: # Why!!???
            raw_data.append(RawData(fields[0], float(fields[1]), float(fields[2]), float(fields[3]), float(fields[4]), float(fields[5]), float(fields[6])))
    sorted_data = sorted(raw_data, key=lambda x: x.date)
    print("got %s records." % len(sorted_data))
    return sorted_data
