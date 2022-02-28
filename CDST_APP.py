#!/usr/bin/env python
# coding: utf-8

# In[1]:


from pywebio.output import put_text
from pywebio.input import *
from pywebio.output import *
from pywebio import *
from pywebio.input import input, FLOAT, TEXT

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
# import seaborn as sns
# import matplotlib.pyplot as plt
# import sklearn

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import plot_confusion_matrix

from numpy import loadtxt
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import InputLayer

from heapq import nlargest
import io


# In[ ]:


# In[ ]:





# In[ ]:





# In[2]:


def readFile(diag_fn, labs_fn, core_fn, adm_fn):
    diag_file = open(diag_fn, encoding='utf-8-sig')
    labs_fn = open(labs_fn, encoding='utf-8-sig')
    core_file = open(core_fn, encoding='utf-8-sig')
    adm_file = open(adm_fn, encoding='utf-8-sig')
    
    diagnosis_admissions_table = pd.read_table(diag_fn)
    
    labs_table = pd.read_table(labs_fn)
    labs_table['PatientID'] =  labs_table['PatientID'] # + '_' + labs_table['AdmissionID'].astype(str)
            
    core_table = pd.read_table(core_file)
    
    adm_table = pd.read_table(adm_file)
    
    return diagnosis_admissions_table, labs_table, core_table, adm_table


# In[3]:


diag_table, labs_table, core_table, adm_table = readFile('AdmissionsDiagnosesCorePopulatedTable_100_new.txt', 'LabsCorePopulatedTable_100_new.txt', 'PatientCorePopulatedTable_100_new.txt', 'AdmissionsCorePopulatedTable_100_new.txt')


# In[4]:


labs_table = labs_table.drop('AdmissionID', axis=1)


# In[5]:


grouped_labs = labs_table.groupby(['PatientID', 'LabName']).aggregate(np.mean)


# In[6]:


labs_dict = {'pat 1':[1, 2, 3]}


# In[7]:


for label, value in grouped_labs.iterrows():

    patient_id = label[0]
    
    if patient_id in labs_dict.keys():
        # print(patient_id + ' already exists')
        labs_dict[patient_id].append(value[0])
    else:
        
        labs_dict[patient_id] = []
        labs_dict[patient_id].append(value[0])


# In[8]:


len(labs_dict.keys())


# In[ ]:





# In[ ]:





# In[ ]:


# from datetime import datetime, timedelta
import datetime

# 1 year has 52 weeks, so we create a delta of 2 years with 2*52
delta = datetime.timedelta(weeks=2*52, days=3, seconds=34)

datetime.timedelta(days=731, seconds=34)
def timedelta_to_years(delta: datetime.timedelta) -> float:
    seconds_in_year = 365.25*24*60*60
    return delta.total_seconds() / seconds_in_year

timedelta_to_years(delta)


from datetime import datetime, timedelta

d = datetime.today() - pd.to_datetime(core_table.loc[0][2])
d = timedelta_to_years(d)
print(d)


# In[9]:


def show_instructions():
    
    popup('Welcome.', [
    put_html('Please download an Electronic Health Record Template for CBC, Metabolic, and Urinalysis Lab Results. Once you upload your patient\'s records, you will receive a diagnosis suggestion based on 6 patients with the most similar medical history.'),
        
    put_file(name='CBC Lab Results', content=b'Absolute Lymphocytes (%):\nAbsolute Neutrophil (%):\nBasophils (k/cumm):\nEosinophils (k/cumm):\nHematocrit (%):\nHemoglobin (gm/dl):\nLymphocytes (k/cumm):\nMCH (pg):\nMCHC (g/dl):\nMean Corpuscular Volume (fl):\nMonocytes (k/cumm):\nNeutrophils (k/cumm):\nPlatelet Count (k/cumm):\nRDW (%):\nRed Blood Cell Count (m/cumm):\nWhite Blood Cell Count (k/cumm):', label='Download CBC Lab Results Template'),
    put_file(name='Metabolic Lab Results', content=b'Albumin (gm/dL):\nALK PHOS (U/L):\nALT/SGPT (U/L):\nANION GAP (mmol/L):\nAST/SGOT (U/L):\nBILI Total (mg/dL):\nBUN (mg/dL):\nCalcium (mg/dL):\nCarbon Dioxide (mmol/L):\nChloride (mmol/L):\nCreatinine (mg/dL):\nGlucose (mg/dL):\nPotassium (mmol/L):\nSodium (mmol/L):\nTotal Protein (gm/dL):', label='Download Metabolic Lab Results Template'),
    put_file(name='Urinalysis Lab Results', content=b'pH:\nRed Blood Cells (rbc/hpf):\nSpecific Gravity:\nWhite Blood Cells (wbc/hpf):', label='Download Urinalysis Lab Results Template'),

    put_button('Got it!', onclick=close_popup)])


# In[10]:


# VARIABLES

cbc_results_file = None
metab_results_file = None
uri_results_file = None

file_names = []
scores_dict = []

# In[11]:


# ---- NOT USING RN ----

def show_upload_tests(btn_val):
    
    if btn_val == 'CBC':
        f = file_upload("Patient's CBC Lab Results:", accept="txt/*", multiple=False)
        cbc_results_file = f['filename']
        # print('... uploading cbc...')
        
    if btn_val == 'Metabolic':
        f = file_upload("Patient's Metabolic Lab Results:", accept="txt/*", multiple=False)['filename']
        metab_results_file = f['filename']
        # print('... uploading metab...')
        
    if btn_val == 'Urinalysis':
        f = file_upload("Patient's Urinalysis Lab Results:", accept="txt/*", multiple=False)['filename']
        uri_results_file = f['filename']
        # print('... uploading uri...')
        


# In[12]:




def read_labs(cbc_labs_fn, metabolic_labs_fn, urinalysis_labs_fn): 
    cbc_labs = []
    metabolic_labs = []
    urinalysis_lab = []
    
    if cbc_labs_fn is not None:
        contents = io.StringIO(cbc_labs_fn['content'].decode("utf-8"))
        for line in contents:
            # print(line.split(':')[1]) # lab value
            string_val = line.split(':')[1].strip()
            cbc_labs.append(float(string_val))
            
    if metabolic_labs_fn is not None:
        contents = io.StringIO(metabolic_labs_fn['content'].decode("utf-8"))
        for line in contents:
            # print(line.split(':')[1]) # lab value
            string_val = line.split(':')[1].strip()
            metabolic_labs.append(float(string_val))            

    if urinalysis_labs_fn is not None:
        contents = io.StringIO(urinalysis_labs_fn['content'].decode("utf-8"))
        for line in contents:
            # print(line.split(':')[1]) # lab value
            string_val = line.split(':')[1].strip()
            urinalysis_lab.append(float(string_val)) 
    
    print("Values read. cbc {} met {} uri {}".format(cbc_labs, metabolic_labs, urinalysis_lab))
    return cbc_labs, metabolic_labs, urinalysis_lab


# In[13]:


def combine_all_labs(cbc_labs, metab_labs, uri_labs):
    
    all_labs = []
    
    for elem in cbc_labs:
        all_labs.append(elem)
        
    for elem in metab_labs:
        all_labs.append(elem)
        
    for elem in uri_labs:
        all_labs.append(elem)
        
    return all_labs


# In[14]:


def compare_similar(new_patient_labs):

    sim_scores = {}

    for patient in labs_dict:
          
        db_pat_labs = np.array(labs_dict[patient])
        new_pat_labs = np.array(new_patient_labs)
        
        
        if(len(db_pat_labs) == len(new_pat_labs)):
            sim_score = 100*(1 - np.mean(abs(db_pat_labs-new_pat_labs)/(db_pat_labs+new_pat_labs)))
            sim_scores[patient] = sim_score
        
       
        # --- sim_score = np.mean(abs(db_pat_labs-new_pat_labs))
        # sim_score = 1 - np.mean(abs((np.array(labs_dict[patient]) - np.array(new_patient_labs))))
        # --- sim_scores[patient] = sim_score
        # print(len(labs_dict[patient]))
        # print(len(new_patient_labs))
        
        # print(sim_score)
        
    # return Nmaxelements(sim_scores, 5) # 5 most similar scores
    return sim_scores
        


# In[15]:




def show_upload_options():
    # return value from file_upload is a list of dict i.e [{'filename':'', 'content':'', 'size': ''}, {}..{}]
    # so just return that value, and we have everything needed.
    upload_data = file_upload("Upload labs:", accept="text/*", multiple=True)
    
    cbc_results = metab_results = uri_results = None
    for metadata in upload_data:
        if 'cbc' in metadata['filename'].lower():
            cbc_results = metadata.copy()
        if 'metab' in metadata['filename'].lower():
            metab_results = metadata.copy()   
        if 'uri' in metadata['filename'].lower():
            uri_results = metadata.copy()
    
    return cbc_results, metab_results, uri_results   


# In[16]:


x_outer = None
y_outer = None

mod = None
x_1 = None
y_1 = None

new_x_outer = None
new_y_outer = None
x_dub_outer = None


# In[17]:


def help_decide():

    disease = input('What condition do you predict your patient has?')
    print(disease)
    
    diagnosis_admissions_table = pd.read_table('AdmissionsDiagnosesCorePopulatedTable_100_new.txt')
    diagnosis_admissions_table['PatientID'] =  diagnosis_admissions_table['PatientID'] + '_' + diagnosis_admissions_table['AdmissionID'].astype(str)
    
    new_labs_table = pd.read_table('LabsCorePopulatedTable_100_new.txt')
    new_labs_table['PatientID'] =  new_labs_table['PatientID'] + '_' + new_labs_table['AdmissionID'].astype(str)
    
    
    diagnosis_admissions_table.sort_values('PatientID', inplace=True)
    total_diagnoses_specific = diagnosis_admissions_table['PrimaryDiagnosisDescription'] # for with details (ex: Protozoal diseases complicating pregnancy, first trimester)

    new_grouped_labs = new_labs_table.groupby(['PatientID', 'LabName']).aggregate(np.mean)
    og_labs = new_grouped_labs.reset_index()
    
    new_x = og_labs.pivot(index='PatientID', columns='LabName', values='LabValue').sort_index()

    
    new_y = []

    for diagnosis in total_diagnoses_specific:
        if disease in diagnosis:
            new_y.append(True)
        else: 
            new_y.append(False)
            
    new_x.fillna(-100, inplace=True)

    new_x_outer = new_x
    new_y_outer = new_y
    
    x_train_lr, x_test_lr, y_train_lr, y_test_lr = train_test_split(new_x, new_y, test_size=.2)
    model_lr = LogisticRegression()
    model_lr.fit(x_train_lr, y_train_lr)
    score_lr = model_lr.score(x_test_lr, y_test_lr)
    # print(model_lr.score(x_test_lr, y_test_lr))

    x_train_knn, x_test_knn, y_train_knn, y_test_knn = train_test_split(new_x, new_y, test_size=.2)
    model_knn = KNeighborsClassifier(n_neighbors = 6)
    model_knn.fit(x_train_knn, y_train_knn)
    score_knn = model_knn.score(x_test_knn, y_test_knn)

    x_train_dt, x_test_dt, y_train_dt, y_test_dt = train_test_split(new_x, new_y, test_size=.2)
    model_dt = DecisionTreeClassifier(max_depth=1)
    model_dt.fit(x_train_dt, y_train_dt)
    score_dt = model_dt.score(x_test_dt, y_test_dt)
    
    model_1 = Sequential()
    model_1.add(InputLayer(input_shape=(35, )))
    model_1.add(Dense(4, activation = 'sigmoid'))
    model_1.add(Dense(4, activation = 'sigmoid'))
    model_1.add(Dense(1, activation = 'sigmoid', use_bias=True))
    model_1.compile(loss='mean_squared_error',
                optimizer = 'adam', 
                metrics = ['accuracy'])
    history = model_1.fit(x_train_lr, np.array(y_train_lr)*1, epochs=10, validation_data=(x_test_lr, np.array(y_test_lr)*1))
    score_nn = history.history['val_accuracy'][0]
    
    x_dub = new_x
    x_dub.drop(x_dub.index, inplace=True) # clearing df
    
    labs_names = list(x_dub)
    
    new_pat_labs_1 = {}
    counter = 0
    # total_pat_labs
    
    for lab_name in labs_names:
        new_pat_labs_1[lab_name] = total_pat_labs[counter]
        counter+=1
        
    x_dub = x_dub.append(new_pat_labs_1, ignore_index=True)
    lr_predict = model_lr.predict(x_dub)[0]
    knn_predict = model_knn.predict(x_dub)[0]
    dt_predict = model_dt.predict(x_dub)[0]
    nn_predict = model_1.predict(x_dub)[0][0]
    nn_final_predict = False
    if nn_predict >= 0.5:
        nn_final_predict = True
    
    x_dub_outer = x_dub
    
    # print(model_lr.predict(x_dub)[0])
    
    with use_scope('scope', clear=True):
        # put_text(model_lr.predict(x_dub)[0])
        
        put_table([
            ['', 'Logistic Regression', 'KNN', 'Decision Tree', 'Neural Network'],
            ['Model Accuracy', round(score_lr, 2), round(score_knn, 2), round(score_dt, 2), round(score_nn, 2)],
            ['Model Prediction', lr_predict, knn_predict, dt_predict, nn_final_predict]
        ])
        
    # plot_confusion_matrix(model_lr, x_test_lr, y_test_lr)
    mod = model_lr
    x_1 = x_test_lr
    y_1 = y_test_lr
        
        
    # put_text(model_lr.predict_proba(x_dub))
    
    val1 = model_lr.predict_proba(x_dub)[0][0]
    val2 = model_lr.predict_proba(x_dub)[0][1]
    
    # if val1 > val2:
        # put_text(val1)
    # if val2 > val1:
        # put_text(val2)
    


# In[18]:


def make_scope():
    with use_scope('scope', clear=True):
        print('created scope')


# In[ ]:


top_6_diags = []


# In[ ]:





# In[ ]:





# In[19]:


def show_info(index, most_sim_patients, infos, rec):
    
    with use_scope('scope', clear=True): # enter the existing scope and clear the
        
        if(rec == ""):
            rec = 'No diagnosis recorded.'
        
        core_index = list(core_table[core_table['PatientID'] == str(most_sim_patients[index])].index)[0]
        from datetime import datetime, timedelta
        d = datetime.today() - pd.to_datetime(core_table.loc[core_index][2])
        d = int(timedelta_to_years(d))
        print(d)
        
        # d = datetime.today() - pd.to_datetime(core_table.loc[core_index][2])
        # d = timedelta_to_years(d)
        # print(d)
        
        
        put_table([
            ['Patient ID', most_sim_patients[index]],
            ['Similarity', str(round(scores_dict[most_sim_patients[index]], 2)) + '%'],
            ['Race, Gender', infos[index]],
            ['Date of Birth', core_table.loc[core_index]['PatientDateOfBirth']],
            ['Age', str(d)],
            ['Marital Status', core_table.loc[core_index]['PatientMaritalStatus']],
            ['Language', core_table.loc[core_index]['PatientLanguage']],
            ['Population % Below Poverty', str(core_table.loc[core_index]['PatientPopulationPercentageBelowPoverty']) + '%'],
            ['Doctor\'s Previous Diagnosees', rec]
        ])
        
        # put_text(most_sim_patients[index])
        # put_text(str(scores_dict[most_sim_patients[index]]) + '% similar')
        # put_text(infos[index])
        # put_text(rec)
        
        #put_code(most_sim_patients[2] + ' - ' + str(scores_dict[most_sim_patients[2]]) + '% similarity\n' + infos[2] + '\n' + recs3)


# In[ ]:





# In[20]:


def main_app():
        # input('What condition do you predict your patient has?')
    global scores_dict
    global total_pat_labs

    # global cbc_labs
    # cbc_labs = []
    # global metabolic_labs
    # metabolic_labs = []
    # global urinalysis_lab
    # urinalysis_lab = []
    
    show_instructions()
    
    # txt = put_text('Please click on a lab test to enter specific results. Once you submit these values, you will receive a diagnosis suggestion based on 6 patients with the most similar medical history.')    

    # put_buttons(['CBC', 'Metabolic', 'Urinalysis'], onclick=show_upload_tests)
    cbc, metab, uri = show_upload_options()
    
    put_button('Help me decide if my diagnosis is reliable.', onclick=help_decide, color='secondary')
    # put_button('HELP', onclick=help_dec, color='secondary')
    
    # read_labs(cbc, metab, uri)
    cbc_labs, metabolic_labs, urinalysis_lab = read_labs(cbc, metab, uri)
    all_new_pat_labs = combine_all_labs(cbc_labs, metabolic_labs, urinalysis_lab)
    total_pat_labs = all_new_pat_labs
    scores_dict = compare_similar(all_new_pat_labs)
    # print(scores_dict)
    
    most_sim_patients = nlargest(6, scores_dict, key = scores_dict.get)
    # print(most_sim_patients)
    
    
    doc_recoms = {}
    infos = []
    
    put_markdown('**You see can patients with the highest similarity below.**')
    
    for pat_id in most_sim_patients:
        
        core_index = list(core_table[core_table['PatientID'] == str(pat_id)].index)[0]
        
        recoms = []
        
        infos.append(core_table.loc[core_index]['PatientRace'] + ', ' + core_table.loc[core_index]['PatientGender'])
        
        diag_indexes = list(diag_table[diag_table['PatientID'] == str(pat_id)].index)
        
        
        for index in diag_indexes:
            
            rec = diag_table.loc[index]['PrimaryDiagnosisDescription']
            recoms.append(rec)
        
        doc_recoms[pat_id] = recoms
        
        
        
    
    recs1 = ""
    recs2 = ""
    recs3 = ""
    recs4 = ""
    recs5 = ""
    recs6 = ""
    
    for pat_id in doc_recoms:
        
        for i in range(5):
        
            if pat_id == most_sim_patients[i]:
                for rec in doc_recoms[pat_id]:
                    if i==0: 
                        recs1 += rec + '\n'
                        top_6_diags.append(rec)
                    if i==1: 
                        recs2 += rec + '\n'
                        top_6_diags.append(rec)
                    if i==2: 
                        recs3 += rec + '\n'
                        top_6_diags.append(rec)
                    if i==3: 
                        recs4 += rec + '\n'
                        top_6_diags.append(rec)
                    if i==4: 
                        recs5 += rec + '\n'
                        top_6_diags.append(rec)
                    if i==5: 
                        recs6 += rec + '\n'
                        top_6_diags.append(rec)
    
    
    put_column([
        put_row([
            put_button('Patient with ' + str(round(scores_dict[most_sim_patients[0]], 4)) + '% similarity', onclick=(lambda: show_info(0, most_sim_patients, infos, recs1)), color='info'), None, # show_info(0, most_sim_patients, infos, recs1), color='info')
            put_button('Patient with ' + str(round(scores_dict[most_sim_patients[1]], 4)) + '% similarity', onclick=(lambda: show_info(1, most_sim_patients, infos, recs2)), color='info'), None,
            put_button('Patient with ' + str(round(scores_dict[most_sim_patients[2]], 4)) + '% similarity', onclick=(lambda: show_info(2, most_sim_patients, infos, recs3)), color='info')

        ])
    ])
    
    put_column([
        put_row([
            put_button('Patient with ' + str(round(scores_dict[most_sim_patients[3]], 4)) + '% similarity', onclick=(lambda: show_info(3, most_sim_patients, infos, recs4)), color='info'), None,
            put_button('Patient with ' + str(round(scores_dict[most_sim_patients[4]], 4)) + '% similarity', onclick=(lambda: show_info(4, most_sim_patients, infos, recs5)), color='info'), None,
            put_button('Patient with ' + str(round(scores_dict[most_sim_patients[5]], 4)) + '% similarity', onclick=(lambda: show_info(5, most_sim_patients, infos, recs6)), color='info')
        ])
    ])
   
    make_scope()

    


# In[21]:


if __name__ == '__main__':
    
    start_server(main_app, port=80, max_payload_size='2M', debug=True)


# In[ ]:


# recs6


# In[ ]:


# core_table.loc[9]['PatientRace']


# In[ ]:


diag_table['PrimaryDiagnosisDescription']


# In[ ]:


diag_table


# In[ ]:


grouped_labs


# In[ ]:


x_outer


# In[ ]:


x_outer


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





