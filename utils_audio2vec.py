import numpy as np
import pandas as pd
import librosa,os
import torch
from common import iterbrowse
from transformers import Data2VecAudioForCTC,Wav2Vec2Processor

train_dir = 'wavdata/ADReSSo21-diagnosis-train/audio'
test_dir = 'wavdata/ADReSSo21-diagnosis-test/audio'

train_filepath = iterbrowse(train_dir)


# load model and processor
processor = Wav2Vec2Processor.from_pretrained("data2vec-audio-base-960h")
model = Data2VecAudioForCTC.from_pretrained("data2vec-audio-base-960h")



def getTrainEmbedding(train_audio_dir,todir,cut_due,sr):
    total_cut_num = 0
    train_filepath = iterbrowse(train_audio_dir)

    for filep in train_filepath:
        print(filep)
        # load wave data
        start = 0

        _, filename = os.path.split(filep)

        labelid = filename.replace('.wav', '')

        if 'cn' in filep:
            label = 0
        else:
            label = 1

        # load wavedata for 16000 sample rating
        x_raw, _ = librosa.load(filep)
        total_duration = librosa.get_duration(y=x_raw, sr=sr)
        print('total time is %s' % (total_duration))
        cut_number = int(total_duration // (cut_due - 1))  # 取得切分段数
        total_cut_num += cut_number

        embedding = []
        for i in range(cut_number):

            end = start + cut_due

            if i == cut_number - 1:
                x_cut = x_raw[start * sr:int(total_duration) * sr]
                print('%s/%s\t%s\t%s' % (i + 1, cut_number, start, total_duration))
            else:
                x_cut = x_raw[start * sr:end * sr]  # cut the audio for every 10s
                print('%s/%s\t%s\t%s' % (i + 1, cut_number, start, end))
            # tokenize
            input_values = processor(x_cut, return_tensors="pt", padding="longest",
                                     sampling_rate=sr).input_values  # Batch size 1

            # retrieve last hidden state
            output = model(input_values, output_hidden_states=True)
            last_hidden_states = output.hidden_states[-1]
            hidden_shape = last_hidden_states.shape

            X_cut = last_hidden_states.reshape(hidden_shape[1], hidden_shape[2]).sum(dim=0)
            Y_cut = torch.tensor(label, dtype=torch.float32)

            data_cut = torch.hstack([X_cut, Y_cut])
            embedding.append(data_cut)

            start = end - 1

        embedding = torch.vstack(embedding)
        print('one audio embedding shape', embedding.shape)

        new_filename = labelid + '.pt'
        torch.save(embedding, os.path.join(todir, new_filename))

    print('total cut num is ', total_cut_num)




def getTestEmbedding(test_audio_dir,labelpath,todir,cut_due,sr):
    total_cut_num = 0

    labels =  pd.read_csv(labelpath)
    label_dic = dict(zip(labels['ID'], labels['Dx']))

    test_filepath = iterbrowse(test_audio_dir)
    for filep in test_filepath:
        print(filep)
        # load wave data
        start = 0

        _, filename = os.path.split(filep)

        labelid = filename.replace('.wav','')
        label = label_dic[labelid]
        if label == 'Control':
            label = 0
        else:
            label = 1

        # load wavedata for 16000 sample rating
        x_raw, _ = librosa.load(filep)
        total_duration = librosa.get_duration(y=x_raw, sr=sr)
        print('total time is %s' % (total_duration))
        cut_number = int(total_duration // (cut_due - 1))  # 取得切分段数
        total_cut_num += cut_number

        embedding = []
        for i in range(cut_number):

            end = start + cut_due

            if i == cut_number - 1:
                x_cut = x_raw[start * sr:int(total_duration) * sr]
                print('%s/%s\t%s\t%s' % (i + 1, cut_number, start, total_duration))
            else:
                x_cut = x_raw[start * sr:end * sr]  # cut the audio for every 10s
                print('%s/%s\t%s\t%s' % (i + 1, cut_number, start, end))
            # tokenize
            input_values = processor(x_cut, return_tensors="pt", padding="longest",
                                     sampling_rate=sr).input_values  # Batch size 1

            # retrieve last hidden state
            output = model(input_values, output_hidden_states=True)
            last_hidden_states = output.hidden_states[-1]
            hidden_shape = last_hidden_states.shape

            X_cut = last_hidden_states.reshape(hidden_shape[1], hidden_shape[2]).sum(dim=0)
            Y_cut = torch.tensor(label,dtype=torch.float32)

            data_cut = torch.hstack([X_cut,Y_cut])
            embedding.append(data_cut)


            start = end - 1

        new_filename = labelid+'.pt'

        embedding = torch.vstack(embedding)
        print('one audio embedding shape',embedding.shape)
        torch.save(embedding, os.path.join(todir, new_filename))

    print('total cut num is ', total_cut_num)

if __name__ == '__main__':
    cut_due = 15
    sr = 16000


    train_audio_dir = 'wavdata/ADReSSo21-diagnosis-train/audio'
    train2path = 'audio-embedding/AD2021_PAR+INV/%ssec/train'%(cut_due)
    if not os.path.exists(train2path):
        os.makedirs(train2path)

    getTrainEmbedding(train_audio_dir,train2path,cut_due,sr)


    test_audio_dir = 'wavdata/ADReSSo21-diagnosis-test/audio'
    label_path = 'wavdata/task1.csv'
    test2path = 'audio-embedding/AD2021_PAR+INV/%ssec/test'%(cut_due)
    if not os.path.exists(test2path):
        os.makedirs(test2path)
    getTestEmbedding(test_audio_dir,label_path,test2path,cut_due,sr)
