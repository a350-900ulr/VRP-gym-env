only = 'chcek'

import argparse

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('-action', type=str, required=False)

	arguments = parser.parse_args()

	only = arguments.action
	print(only)
