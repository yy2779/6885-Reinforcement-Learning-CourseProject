import csv
import numpy as np
import math


def read_data(filename):
    with open(filename, 'r') as csvfile:
        reader = csv.reader(csvfile)
        hotel_info = []
        poi_info = []
        next(reader, None)
        for line in reader:
            if "hotel" in line[1]:
                hotel_info.append([line[0], line[1], float(line[2]), float(line[3]), float(line[4]), float(line[5])])
            elif "poi" in line[1]:
                poi_info.append([line[0], line[1], float(line[2]), float(line[3]), float(line[4]), float(line[5])])
    return hotel_info, poi_info


def distance(location1, location2):
    C = math.sin(location1[0]) * math.sin(location2[0]) + math.cos(location1[0]) * math.cos(location2[0]) * math.cos(location1[1] - location2[1])
    C = np.clip(C, -1., 1.)
    dis = 6371.004 * math.acos(C) * math.pi / 180
    return dis


class tour_env():
    def __init__(self):
        # print('Construct Environment: Tour Maker')
        self.hotel_info, self.poi_info = read_data('data.csv')
        self.hotel_num = len(self.hotel_info)
        self.poi_num = len(self.poi_info)

        # Total time steps
        # self.init_time = 8  # start from 8am. 1 hour for lunch and 1 hour for dinner
        self.total_time_step = 10
        self.time = -1
        self.counter = -1
        self.exceed_counter = -1

        # Total number of actions: the number of poi + 1, and 0 represent go back to hotel
        self.nA = self.poi_num + 1
        # Total number of states
        self.nS = self.hotel_num * (self.nA ** (self.total_time_step + 1) - 1) // self.poi_num

        self.history = []  # list, used to save history
        self.his_table = np.zeros([self.poi_num]).astype(np.int32)
        self.hotel = None  # used to remember the init hotel index
        self.state = -1  # used to save state
        self.place_info = -1  # help to save state, save the current location

    def reset(self, init_hotel_input=None):
        # print('reset the environment')
        if init_hotel_input is None:
            init_hotel = np.random.randint(0, self.hotel_num)  # random choose a hotel for initialization
        else:
            if init_hotel_input not in range(self.hotel_num):  # choose the given hotel for initialize
                # print('Illegal init hotel')
                raise NameError
            init_hotel = init_hotel_input  # use the pre-set hotel

        self.state = init_hotel  # init state/hotel
        self.hotel = init_hotel  # remember the init hotel index
        self.place_info = self.hotel_info[init_hotel]
        self.history = []  # reset history list
        self.his_table = np.zeros([self.poi_num]).astype(np.int32)
        self.history.append([0, init_hotel])
        self.time = 0  # reset time
        self.counter = 0  # reset counter
        self.exceed_counter = 0
        return (self.state, self.time, self.place_info[3], self.place_info[4], self.place_info[2], self.place_info[5], self.his_table)
        # return self.state
        # state, current time, location, location, recommend time, rate, have spent times

    def reward(self, place_info, next_place_info, transfer_time):
        reward = -1 * transfer_time  # calculate the transfer reward
        # print('transfer penalty, R = ', -2 * transfer_time)
        if 'poi' in next_place_info[1]:  # calculate the reward for poi
            count = 0
            for i in range(1, self.counter + 1):  # count how many times have visited.
                if str(self.history[i][1]) in next_place_info[1]:
                    count += 1
            # print('count: ', count)
            if count < 1:
                reward += 10 * next_place_info[5]
                # print(10 * next_place_info[5])
            else:
                reward += 10 * next_place_info[5] * (next_place_info[2] - count) / (next_place_info[2] - 1)
                # print(10 * next_place_info[5] * (next_place_info[2] - count) / next_place_info[2])
        else:  # calculate reward for hotel, only penalty for late
            if self.time > self.total_time_step:
                reward -= ((self.time - self.total_time_step) ** 2) * 5
                # print('Late penalty, R = ', -10 * (self.time - self.total_time_step))
        return reward

    def step(self, action):
        # print('')
        # print('current time: ', self.time)
        place_info = self.place_info
        now_location = [place_info[3], place_info[4]]
        done = False

        # avoid illegal action
        if action not in range(self.nA):
            # print('Illegal action')
            raise NameError

        if action == 0:  # go back to hotel
            # print('choose hotel')
            next_place_info = self.hotel_info[self.hotel]
        else:  # go to a poi
            # print('choose poi', action)
            next_place_info = self.poi_info[action - 1]
        next_location = [next_place_info[3], next_place_info[4]]

        if next_place_info[1] == place_info[1]:  # stay in same place
            # print('stay in same place')
            transfer_time = 0
        else:  # go to different place
            transfer_time = (10 + 2 * distance(now_location,
                                               next_location)) // 30 + 1  # calculate the time needed to transfer
            # print('go to different place, time for transfer: ', transfer_time)

        next_time = self.time + 1 + transfer_time  # update time counter

        if next_time >= self.total_time_step:
            # print('exceed time limit, not allowed, re-action', self.exceed_counter)
            self.exceed_counter += 1
            reward = 0
            info = True

            # try to find a place that won't exceed time limit for 3 times, or go back to hotel
            if self.exceed_counter > 2:
                info = False
                done = True
                # print('can not find next place, go back to hotel')
                next_state = (self.state - self.hotel_num * (
                self.nA ** self.counter - 1) // self.poi_num) * self.nA + self.hotel_num * (
                self.nA ** (self.counter + 1) - 1) // self.poi_num + 0
                self.state = next_state
                # calculate the time for go back to hotel
                next_place_info = self.hotel_info[self.hotel]
                next_location = [next_place_info[3], next_place_info[4]]
                transfer_time = (10 + 2 * distance(now_location,
                                                   next_location)) // 30 + 1  # calculate the time needed to transfer
                self.time += 1 + transfer_time
                reward = self.reward(place_info, next_place_info, transfer_time)
                # print('back to hotel, time for transfer: ', transfer_time)
                self.history.append([self.time, -1])  # set place as -1 to represent end
            # return self.state, reward, done, info
            return (self.state, self.time, next_place_info[3], next_place_info[4], next_place_info[2], next_place_info[5], self.his_table), reward, done, info
            # state, current time, location, location, recommend time, rate, have spent times
        else:
            # print('change to next place within time limit')
            info = False
            self.exceed_counter = 0
            reward = self.reward(place_info, next_place_info, transfer_time)
            next_state = (self.state - self.hotel_num * (self.nA ** self.counter - 1) // self.poi_num) * self.nA + self.hotel_num * (self.nA ** (self.counter + 1) - 1) // self.poi_num + action
            self.counter += 1
            self.state = next_state
            self.time = next_time
            self.history.append([self.time, action])  # update history
            self.place_info = next_place_info  # update place information
            if action > 0:
                self.his_table[action-1] += 1
            # return self.state, reward, done, info
            return (self.state, self.time, next_place_info[3], next_place_info[4], next_place_info[2], next_place_info[5], self.his_table), reward, done, info
            # state, current time, location, location, recommend time, rate, have spent times
