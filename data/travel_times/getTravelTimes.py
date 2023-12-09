# acquires the travel times between all places

inputFile = '../places/attractions.txt'
outputFile = 'wien_travel_times.csv'
depart = '2023-06-14 11:00'  # set all queries to be at the same time of day so that the results are fair
offset = [0, 0]  # incase it crashes before finishing, to continue writing to file


import pandas as pd
import datetime      # to read input date parameter
import time          # to convert input date to unix time for google
import urllib.parse  # to convert address to URL safe format
import requests      # to make google maps API call
import json

# get each place name & address into python lists
places = pd.read_csv(inputFile, sep=';')
names = places['name'].tolist()
addresses = places['address'].tolist()

errors = []  # will hold any failed requests

# convert depart date into unix time for google
depart_unix = str(int(time.mktime(
	datetime.datetime.strptime(depart, '%Y-%m-%d %H:%M').timetuple()
)))

# to reduce the number of API calls the distance between 2 places is only calculated in 1 direction

# 80 places, combinations = 79 + 78 + 77 + 76....
# falling summation = (n^2 - n) / 2 = 3160 combinations
# combinations * 4 modes of transport = 12640 queries total
for place1index in range(len(places)):
	if place1index <= offset[0]:
		continue
	for place2index in range(place1index+1, len(places)):
		if place2index <= offset[1]:
			continue
		# progress bar
		print(f'[{place1index},{place2index}]', end='')

		for mode in ['walking', 'transit', 'bicycling', 'driving']:
			# build google maps url & request GET
			response = requests.request('GET',
				'https://maps.googleapis.com/maps/api/distancematrix/json'
				'?key=' + open('data/get_location_data/api_key', 'r').readlines()[0] +
				'&origins=' + urllib.parse.quote(addresses[place1index], safe='') +
				'&destinations=' + urllib.parse.quote(addresses[place2index], safe='') +
				'&mode=' + mode +
				'&departure_time=' + depart_unix
			)

			# navigate weird json structure
			info = json.loads(response.text)['rows'][0]['elements'][0]

			# if directions where found, append it to the dataframe
			# otherwise, create an errors list
			if info['status'] == 'OK':
				outputFileWriter = open(outputFile, 'a')
				outputFileWriter.write(
					f"{place1index};{place2index};"
					f"{names[place1index]};"f"{names[place2index]};{mode};"
					f"{info['duration']['value'] / 60};{info['distance']['value']}\n"
				)
				outputFileWriter.close()

			else:
				print(f"unexpected status {info['status']}, see errors list")
				errors.append([place1index, place2index, mode, response])

for error in errors:
	f = open(f'error[{error[0]}, {error[1]}]_{error[2]}.json', 'w')
	f.write(error[3].text)
	f.close()
