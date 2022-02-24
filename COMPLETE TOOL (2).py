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
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn
from sklearn.model_selection import train_test_split 

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import plot_confusion_matrix

from heapq import nlargest


# In[ ]:





# In[ ]:





# In[2]:


def readFile(diag_fn, labs_fn, core_fn):
    diag_file = open(diag_fn)
    labs_fn = open(labs_fn)
    core_file = open(core_fn)
    
    diagnosis_admissions_table = pd.read_table(diag_fn)
    
    labs_table = pd.read_table(labs_fn)
    labs_table['PatientID'] =  labs_table['PatientID'] # + '_' + labs_table['AdmissionID'].astype(str)
            
    core_table = pd.read_table(core_file)
    
    return diagnosis_admissions_table, labs_table, core_table


# In[3]:


diag_table, labs_table, core_table = readFile('AdmissionsDiagnosesCorePopulatedTable_100_new.txt', 'LabsCorePopulatedTable_100_new.txt', 'PatientCorePopulatedTable_100_new.txt')


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





# In[9]:


def show_instructions():
    
    popup('Welcome.', [
    put_html('Please download an Electronic Health Record Template for CBC, Metabolic, and Urinalysis Lab Results. Once you upload your patient\'s records, you will receive a diagnosis suggestion based on 6 patients with the most similar medical history.'),
        
    put_file(name='CBC Lab Results', content=b'Absolute Lymphocytes:\nAbsolute Neutrophil:\nBasophils:\nEosinophils:\nHematocrit:\nHemoglobin:\nLymphocytes:\nMCH:\nMCHC:\nMean Corpuscular Volume:\nMonocytes:\nNeutrophils:\nPlatelet Count:\nRDW:\nRed Blood Cell Count:\nWhite Blood Cell Count:', label='Download CBC Lab Results Template'),
    put_file(name='Metabolic Lab Results', content=b'Albumin:\nALK PHOS:\nALT/SGPT:\nANION GAP:\nAST/SGOT:\nBILI Total:\nBUN:\nCalcium\nCarbon Dioxide:\nChloride:\nCreatinine:\nGlucose:\nPotassium:\nSodium:\nTotal Protein:', label='Download Metabolic Lab Results Template'),
    put_file(name='Urinalysis Lab Results', content=b'pH:\nRed Blood Cells:\nSpecific Gravity:\nWhite Blood Cells:', label='Download Urinalysis Lab Results Template'),

    put_button('Got it!', onclick=close_popup)])


# In[10]:


# VARIABLES

cbc_results_file = None
metab_results_file = None
uri_results_file = None

cbc_labs = []
metabolic_labs = []
urinalysis_lab = []

file_names = []


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
        
    with open(str(cbc_labs_fn)) as cbc_f:
        for line in cbc_f:
            # print(line.split(':')[1]) # lab value
            string_val = line.split(':')[1]
            cbc_labs.append(float(string_val))
            
    with open(metabolic_labs_fn) as metabolic_f:
        for line in metabolic_f:
            # print(line.split(':')[1]) # lab value
            string_val = line.split(':')[1]
            metabolic_labs.append(float(string_val))
            
    with open(urinalysis_labs_fn) as urinalysis_f:
        for line in urinalysis_f:
            # print(line.split(':')[1]) # lab value
            string_val = line.split(':')[1]
            urinalysis_lab.append(float(string_val))


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
    files = file_upload("Upload labs:", accept="text/*", multiple=True)
    for file in files:
        file_names.append(file['filename'])
            
            
    for file in file_names:
        if 'cbc' in file:
            cbc_results_file = file
        if 'metab' in file:
            metab_results_file = file    
        if 'uri' in file:
            uri_results_file = file
    
    return cbc_results_file, metab_results_file, uri_results_file
    print(cbc_results_file + ', ' + metab_results_file + ', ' + uri_results_file)


# In[16]:


x_outer = None


# In[17]:


def help_decide():
    # print('help')
    disease = input('What condition do you predict your patient has?')


# In[18]:


def make_scope():
    with use_scope('scope'):
        print('created scope')


# In[ ]:





# In[ ]:





# In[ ]:





# In[19]:


def show_info(index, most_sim_patients, infos, rec):
    
    with use_scope('scope', clear=True): # enter the existing scope and clear the
        
        put_text(most_sim_patients[index])
        put_text(str(scores_dict[most_sim_patients[index]]) + '% similar')
        put_text(infos[index])
        put_text(rec)
        #put_code(most_sim_patients[2] + ' - ' + str(scores_dict[most_sim_patients[2]]) + '% similarity\n' + infos[2] + '\n' + recs3)


# In[ ]:





# In[ ]:





# In[20]:


if __name__ == '__main__':
    
    # input('What condition do you predict your patient has?')
    
    show_instructions()
    
    txt = put_text('Please click on a lab test to enter specific results. Once you submit these values, you will receive a diagnosis suggestion based on 6 patients with the most similar medical history.')    

    # put_buttons(['CBC', 'Metabolic', 'Urinalysis'], onclick=show_upload_tests)
    cbc, metab, uri = show_upload_options()
    
    put_button('Help me decide if my diagnosis is reliable.', onclick=help_decide, color='secondary')
    # put_button('HELP', onclick=help_dec, color='secondary')
    
    read_labs(cbc, metab, uri)
    
    all_new_pat_labs = combine_all_labs(cbc_labs, metabolic_labs, urinalysis_lab)
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
        # print(pat_id + ':')
        # put_text(pat_id + ': ' + core_table.loc[core_index]['PatientRace'] + ', ' + core_table.loc[core_index]['PatientGender'])
        infos.append(core_table.loc[core_index]['PatientRace'] + ', ' + core_table.loc[core_index]['PatientGender'])
        
        diag_indexes = list(diag_table[diag_table['PatientID'] == str(pat_id)].index)
        
        for index in diag_indexes:
            
            rec = diag_table.loc[index]['PrimaryDiagnosisDescription']
            recoms.append(rec)
            
            # put_text(rec)
            
        # put_text('\n')
        
        doc_recoms[pat_id] = recoms
        
        
        
    
    
    # put_column([
        # put_text(scores_dict[pat_id]),
        # put_text(core_table.loc[core_index]['PatientRace']),
        # put_code(core_table.loc[core_index]['PatientGender'])
    # ])
    
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
                    if i==0: recs1 += rec + '\n'
                    if i==1: recs2 += rec + '\n'
                    if i==2: recs3 += rec + '\n'
                    if i==3: recs4 += rec + '\n'
                    if i==4: recs5 += rec + '\n'
                    if i==5: recs6 += rec + '\n'
    
    put_column([
        put_row([
            put_button('Patient with ' + str(scores_dict[most_sim_patients[0]]) + '% similarity', onclick=(lambda: show_info(0, most_sim_patients, infos, recs1)), color='info'), None, # show_info(0, most_sim_patients, infos, recs1), color='info')
            put_button('Patient with ' + str(scores_dict[most_sim_patients[1]]) + '% similarity', onclick=(lambda: show_info(1, most_sim_patients, infos, recs2)), color='info'), None,
            put_button('Patient with ' + str(scores_dict[most_sim_patients[2]]) + '% similarity', onclick=(lambda: show_info(2, most_sim_patients, infos, recs3)), color='info')

        ])
    ])
    
    put_column([
        put_row([
            put_button('Patient with ' + str(scores_dict[most_sim_patients[3]]) + '% similarity', onclick=(lambda: show_info(3, most_sim_patients, infos, recs4)), color='info'), None,
            put_button('Patient with ' + str(scores_dict[most_sim_patients[4]]) + '% similarity', onclick=(lambda: show_info(4, most_sim_patients, infos, recs5)), color='info'), None,
            put_button('Patient with ' + str(scores_dict[most_sim_patients[5]]) + '% similarity', onclick=(lambda: show_info(5, most_sim_patients, infos, recs6)), color='info')
        ])
    ])
   
    make_scope()

    #put_row([
        #put_code(most_sim_patients[0] + ' - ' + str(scores_dict[most_sim_patients[0]]) + '% similarity\n' + infos[0] + '\n' + recs1)
    #])   
    
    #put_row([
        #put_code(most_sim_patients[1] + ' - ' + str(scores_dict[most_sim_patients[1]]) + '% similarity\n' + infos[1] + '\n' + recs2)
    #])
    #put_row([
        #put_code(most_sim_patients[2] + ' - ' + str(scores_dict[most_sim_patients[2]]) + '% similarity\n' + infos[2] + '\n' + recs3)
    #])
    #put_row([
        #put_code(most_sim_patients[3] + ' - ' + str(scores_dict[most_sim_patients[3]]) + '% similarity\n' + infos[3] + '\n' + recs4)
    #])
    #put_row([
        #put_code(most_sim_patients[4] + ' - ' + str(scores_dict[most_sim_patients[4]]) + '% similarity\n' + infos[4] + '\n' + recs5)
    #])
    #put_row([
        #put_code(most_sim_patients[5] + ' - ' + str(scores_dict[most_sim_patients[5]]) + '% similarity\n' + infos[5] + '\n' + recs6)
    #])
    
    

    


# In[21]:


# recs6


# In[22]:


# core_table.loc[9]['PatientRace']


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




