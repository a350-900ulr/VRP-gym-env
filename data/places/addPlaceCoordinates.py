# adds coordinates & index to places

inputFile = 'attractions.txt'  # name;address
outputFile = 'places.csv'  # index;name;address;latitude;longitude


import googlemaps
import pandas as pd

gmaps = googlemaps.Client(key=open('../api_key', 'r').readlines()[0])
places = pd.read_csv('attractions.txt', sep=';')

x, y = [], []

for address in places['address']:
	coordinates = gmaps.geocode(address)[0]['geometry']['viewport']

	# google maps returns the northeast & southeast coordinates of the locations bounding box
	# for simplicity, an average of these places will be recorded
	x.append((coordinates['northeast']['lat'] + coordinates['southwest']['lat']) / 2)
	y.append((coordinates['northeast']['lng'] + coordinates['southwest']['lng']) / 2)

places['latitude'] = x
places['longitude'] = y

places.to_csv(outputFile, sep=';')
