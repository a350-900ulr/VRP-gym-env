# adds coordinates & index to places

import pandas as pd
import googlemaps

gmaps = googlemaps.Client(key=open('data/get_location_data/api_key', 'r').readlines()[0])

places = pd.read_csv('attractions.txt', sep=';')

x, y = [], []

for address in places['address']:
	geocode_result = gmaps.geocode(address)

	coordinates = geocode_result[0]['geometry']['viewport']

	x.append((coordinates['northeast']['lat'] + coordinates['southwest']['lat']) / 2)
	y.append((coordinates['northeast']['lng'] + coordinates['southwest']['lng']) / 2)

places['latitude'] = x
places['longitude'] = y



places.to_csv('places.csv', sep=';')