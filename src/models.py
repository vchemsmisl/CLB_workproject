import wget

url = 'http://vectors.nlpl.eu/repository/20/213.zip'
filename = wget.download(url, out='/models/geowac/')

