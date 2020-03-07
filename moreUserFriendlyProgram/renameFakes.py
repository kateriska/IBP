import os

path = '/home/katerina/Documents/FinalProgramIBP/allLiveImgNew2/'
i = 90
for filename in os.listdir(path):
    os.rename(os.path.join(path,filename), os.path.join(path,'live'+str(i)+'.jpg'))
    i = i +1
