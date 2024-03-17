import pandas as pd
from pytube import YouTube
import scrapetube
import time

def Download(link,filename, path=''):  
    try:
        youtubeObject = YouTube(link)
        youtubeObject = youtubeObject.streams.get_highest_resolution()
        youtubeObject.download(filename=path+filename)
        # print(youtubeObject.title)
        return youtubeObject.title

    except:
        print("An error has occurred")
        return None

path_dataset = '/home/lara/Documents/LSM-lsmtv/localjobs/row_dataset_new.csv'
path = '/media/lara/Elements/LSMTV/all_videos/'
df = pd.read_csv(path_dataset)
for ind in df.index:
    if df.at[ind, 'downloaded']:
      continue

    print('Downloading {}/{} ({})'.format(ind, len(df.index), df['videoID'][ind]))
    title = Download('https://www.youtube.com/watch?v={}'.format(df['videoID'][ind]), df['videoName'][ind], path=path)
    if title != None:
      df.at[ind, 'downloaded'] = True
      df.at[ind, 'title'] = title

      # %mv *.mp4 '/content/drive/MyDrive/P2MNN/lsmtv/'
      # files.download('/content/' + '"' + title + '"' + '.mp4')
      df.to_csv(path_dataset, index=False)
      
    # break
    # if ind >= 1:
    #   time.sleep(180)
    #   break
