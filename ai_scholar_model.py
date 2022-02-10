# -*- coding: utf-8 -*
"""Untitled38.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1WIHeW63T50zOCwuokr2ve65i23b7bUsV
"""


import pandas as pd
import torch
from torch import nn
from sklearn.model_selection import train_test_split
from transformers import DistilBertModel, DistilBertTokenizer

df = pd.read_csv('dataframe_sample_preproc.csv')
for col in df.columns:
    print(col, df[col].isnull().values.any())

nums = ['year', 'have_published', 'referenceCount', 'has_acronym', 'has_mark', 'has_colon',
       'num_low_freq_words', 'numbers_in_Abs', 'abstract_ratio_passive',
       'abstract_comma_count', 'abstract_num_sent', 'abstract_num_word',
       'abstract_difficult_word_ratio', 'abstract_avg_num_syllable',
       'abstract_flesch_readability', 'abstract_num_types',
       'abstract_MATTR_density', 'abstract_ttr', 'question_in_abstract',
       'question_in_title', 'humor_probability', 'title_length',
       'number_in_title']
a = [ 'History', 'Economics','Chemistry',
 'Engineering',
 'Geology',
 'Mathematics',
 'Sociology',
 'Environmental Science',
 'Materials Science',
 'Philosophy',
 'Physics',
 'Computer Science',
 'Psychology',
 'Political Science',
 'Biology',
 'Art',
 'Geography',
 'Business',
 'Medicine']

fields_of_study = ['History', 'Economics','Chemistry',
 'Engineering',
 'Geology',
 'Mathematics',
 'Sociology',
 'Environmental Science',
 'Materials Science',
 'Philosophy',
 'Physics',
 'Computer Science',
 'Psychology',
 'Political Science',
 'Biology',
 'Art',
 'Geography',
 'Business',
 'Medicine']


fos_occurences = dict()
for field in fields_of_study:
    fos_occurences[field] = []


df['fieldsOfStudy'] = df['fieldsOfStudy'].fillna('[]')

"""
for l in df['fieldsOfStudy'].to_list():
    for field in fields_of_study:
        if field in eval(l):
            fos_occurences[field].append(1)
        else:
            fos_occurences[field].append(0)

for field in fields_of_study:
    df[field] = fos_occurences[field]
"""

df['year'] = df['year'].fillna(2015)
df['have_published'] = df['have_published'].fillna(0)
class Dataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.abstracts = data['abstract'].to_list()
        self.titles = data['title'].to_list()
        self.venues = data['venue'].to_list()
        self.nums = data[nums].to_numpy()
        #print('venues', len(self.venues))
        #print('nums', len(self.nums))

        self.labels = data['citationCount'].to_list()

    def __len__(self):
        return len(self.abstracts)

    def __getitem__(self, index):
        abstract = self.abstracts[index]
        title = self.titles[index]
        venue = self.venues[index]
        numeric = self.nums[index]

        label = self.labels[index]

        return title, abstract, venue, numeric, label

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:2" if use_cuda else "cpu")
params = {'batch_size': 32,
          'shuffle': True,
          'num_workers': 0}

df_train, df_test = train_test_split(df, test_size=0.1)

training_set = Dataset(df_train)
training_generator = torch.utils.data.DataLoader(training_set, **params)

validation_set = Dataset(df_test)
validation_generator = torch.utils.data.DataLoader(validation_set, **params)

class HedgeRegressor(nn.Module):
    def __init__(self):
        super(HedgeRegressor, self).__init__()

        self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        self.abstract_bert = DistilBertModel.from_pretrained('distilbert-base-uncased')
        self.title_bert = DistilBertModel.from_pretrained('distilbert-base-uncased')
        self.venue_bert = DistilBertModel.from_pretrained('distilbert-base-uncased')

        self.abstract_layer = nn.Linear(768, 300)
        self.title_layer = nn.Linear(768, 100)
        self.venue_layer = nn.Linear(768, 30)
        self.l1 = nn.Linear(472-19, 100)
        #self.classifier_head = nn.Sequential(nn.Linear(400,200), nn.ReLU(),nn.Linear(200,50),nn.ReLU(),nn.Linear(50,2))
        self.l2 = nn.Linear(100,2)
        self.dropout = nn.Dropout(p = 0.2)
        self.relu = nn.ReLU()

    def forward(self, title, abstract, venue, num):

        title_tensor = self.tokenizer(list(title), max_length=30, truncation=True, padding='longest', return_tensors='pt').input_ids.to(device)
        abstract_tensor = self.tokenizer(list(abstract), max_length = 300, truncation = True, padding='longest', return_tensors='pt').input_ids.to(device)
        venue_tensor = self.tokenizer(list(venue), max_length = 30, truncation = True, padding='longest', return_tensors='pt').input_ids.to(device)

        title_vec = self.dropout(self.title_bert(title_tensor).last_hidden_state)
        abstract_vec = self.dropout(self.abstract_bert(abstract_tensor).last_hidden_state)
        venue_vec = self.dropout(self.venue_bert(venue_tensor).last_hidden_state)

        title_rep = self.title_layer(title_vec[:,0,:])
        abstract_rep = self.abstract_layer(abstract_vec[:,0,:])
        venue_rep = self.venue_layer(venue_vec[:,0,:])

        text_rep = torch.cat((title_rep, abstract_rep, venue_rep), dim = 1).float()

        numerics = torch.squeeze(num).float()
        #print(numerics.shape)
        #print(text_rep.shape)
        all = self.relu(torch.cat((text_rep, numerics), dim = 1).float())

        res = self.dropout(self.relu(self.l1(all)))

        return self.l2(res)

model = HedgeRegressor().to(device)
loss_fn = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(model.parameters(), lr = 1e-6)

for epoch in range(10):
    i = 0
    model = model.train()

    overall_loss = 0.0
    overall_correct = 0
    
    for titles, abstracts, venues, nums, labels in training_generator:
        i += 1
        if i % 100 == 0:
          print(i)
        output = model(titles, abstracts, venues, nums.to(device))
        loss = loss_fn(output.squeeze(), labels.long().to(device))
        #print('out', output)
        #print('loss', loss)
        loss.backward()
        optimizer.step()

        overall_loss += loss.item()
        #print(torch.argmax(output.squeeze().cpu(), dim=1))
        #print(labels)
        overall_correct += torch.sum(torch.argmax(output.squeeze().cpu(), dim=1) == labels)
            
    print('epoch loss: ' +  str(overall_loss/(i*2*16)))
    print('epoch accuracy: ' +  str(overall_correct/(i*2*16)))
    torch.save(model.state_dict(), f'epoch{epoch}_full_model1.pt')

    with torch.no_grad():
        j = 0
        model = model.eval()
        overall_loss = 0.0
        overall_correct = 0.0

        for titles, abstracts, venues, nums, labels in validation_generator:
            j += 1

            output = model(titles, abstracts, venues, nums.to(device))
            loss = loss_fn(output.squeeze(), labels.long().to(device))

            print('titles', titles)
            print('labels', labels)
            print('preds', torch.argmax(output.squeeze().cpu(), dim=1))

            overall_loss += loss.item()
            overall_correct += torch.sum(torch.argmax(output.squeeze().cpu(), dim=1) == labels)
            
    print('val loss: ' +  str(overall_loss/(j*2*16)))
    print('val accuracy: ' +  str(overall_correct/(j*2*16)))


