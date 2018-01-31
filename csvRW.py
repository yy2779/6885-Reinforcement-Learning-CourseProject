import csv
import numpy as np


def generate_random_data(filename, hotel_num=3, poi_num=10):
    with open(filename, 'w', newline='') as csvfile:
        fieldnames = ['idx', 'rate', 'longitude', 'latitude', 'recommended time']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        # generate hotel
        for i in range(hotel_num):
            hotel_idx = "hotel" + str(i+1)
            hotel_rate = np.random.randint(1,11) # rate from 1 to 10
            hotel_longitude = int(np.random.normal(0, 30))
            hotel_latitude = int(np.random.normal(0, 30))
            writer.writerow({'idx': hotel_idx, 'rate': hotel_rate, 'longitude':hotel_longitude, 'latitude':hotel_latitude})

        # generate poi
        for j in range(poi_num):
            poi_idx = "poi" + str(j+1)
            poi_rate = np.random.randint(1,11) # rate from 1 to 10
            poi_longitude = int(np.random.normal(0, 30))
            poi_latitude = int(np.random.normal(0, 30))
            poi_recommended_time = 0.5 * np.random.randint(1,17)
            writer.writerow({'idx': poi_idx, 'rate': poi_rate, 'longitude':poi_longitude, 'latitude':poi_latitude, 'recommended time': poi_recommended_time})


def read_data(filename):
    with open(filename, 'r') as csvfile:
        reader = csv.reader(csvfile)
        hotel_info = []
        poi_info = []
        next(reader, None)
        for line in reader:
            if "hotel" in line[0]:
                hotel_info.append([line[0], int(line[1]), float(line[2]), float(line[3])])
            elif "poi" in line[0]:
                poi_info.append([line[0], int(line[1]), float(line[2]), float(line[3]), float(line[4])])
    print(hotel_info)
    print(poi_info)
    return hotel_info, poi_info


generate_random_data('data.csv', hotel_num=2, poi_num=3)
hotel_info, poi_info = read_data('data.csv')

