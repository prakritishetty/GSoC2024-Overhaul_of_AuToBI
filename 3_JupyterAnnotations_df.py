import glob
import numpy as np
import pandas as pd
import os
import zipfile

arr=[]

# Path to the ZIP file
zip_file_path = './BURSC_FINAL.zip'

# Directory where you want to extract the contents
extract_dir = './BURSC_FINAL/'

# Create the extraction directory if it doesn't exist
os.makedirs(extract_dir, exist_ok=True)

# Extract the ZIP file
with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
    zip_ref.extractall(extract_dir)

print("Extraction complete!")

directory = './BURSC_FINAL/BURSC_FINAL'

# Iterate through all files in the directory
for filename in glob.glob(directory+ '/*' ):
  # print()
  with open(filename) as f:
    lines = f.readlines()
    id = filename.split('/')[-1]   # identifier for corresponding speech files
    # print(id)
    # color = lines[2].split(' ')[1]

    for idx, line in enumerate(lines):
      if line.startswith(" "): #This eliminates all the header lines from consideration.
        line = line.strip()
        line = line.split(' ')
        timestamp = line[0] #timestamp of occurence of the phrase boundary
        line = line[1:]

        for idx,string in enumerate(line):
          # Issues here are that the no of spaces at the beginning is not fixed (it can be 2 or 3)
          # The break index is also not the last element of the array, because occasionally 5/6 are also specified.
          if(string!=''):
            # print(idx)
            # print(line[idx:])
            line = line[idx:]
            if(1<len(line)):
              index = line[1]
            else:
              index = 0
            break

        #Handling for 3- and 4-
        if(index=='3-'):
          # print("inside")
          index = '3'
        if(index=='4-'):
          index = '4'


        if(index=='3' or index=='4'):
          arr.append( [id, timestamp,index])
        else:
          arr.append( [id, timestamp,'0'])

df = pd.DataFrame(arr, columns = ['id','timestamp','index'])
print(df)

print(df['index'].unique())

df.to_excel( 'bursc_output.xlsx')



