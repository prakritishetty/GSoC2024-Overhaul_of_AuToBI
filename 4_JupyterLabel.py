import glob
import numpy as np
import pandas as pd
import os
import zipfile

df  = pd.read_excel("bursc_output.xlsx")
print(df)

data = pd.read_excel("speech_bursc_output.xlsx")
print(data)

def new_col(row):
    i = row['name']
    
  # data has 2709 rows, and df has 32,244. Hence choose only those break files which have a corresponding frame
    y = df[df['id'] == i.split('.wav')[0] +".BRK"]
    ans=[]
    print(row)

    print(len(y["index"]))
    for k in range (0,len(y["timestamp"])-1):
        if y["timestamp"].iloc[k] > row.end_time:
            break
        while (k<len(y['timestamp']) and y["timestamp"].iloc[k] >= row.start_time) and (y["timestamp"].iloc[k] <= row.end_time):
            if(y["index"].iloc[k]):
                ans.append(y["index"].iloc[k])
            k+=1
            print("ANS", ans)

    if(len(ans)==0):
        ans=[0]

    if(len(ans)>1):
        print("inside len ans if")
        if '3' in ans:
            ans = [3]
        if '4' in ans:
            ans = [4]
        else:
            ans = [0]

    finalans = ans[0]
    print('ANS FINAL', finalans)

    return finalans
  



data['label'] = data.apply(new_col, axis=1)

data.to_excel( 'final_bursc_data.xlsx')
