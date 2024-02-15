import wget
import os
import zipfile
from pathlib import Path


url = 'http://vectors.nlpl.eu/repository/20/213.zip'
if not os.listdir('C:\pyproj\CLB_workproject\models'):
    filename = wget.download(url, out='C:\pyproj\CLB_workproject\models')
# else:

filename = Path(f"C:\pyproj\CLB_workproject\models\{213}.zip")

geowac_path = Path("C:\pyproj\CLB_workproject\models\geowac")

if not os.path.exists(geowac_path):
    os.makedirs(geowac_path)

with zipfile.ZipFile(filename, 'r') as zip_file:
    zip_file.extractall(geowac_path)


