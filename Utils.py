#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 18 06:25:11 2018

@author: seb
"""

import uuid
from random import shuffle
from keras.preprocessing.image import ImageDataGenerator, image
import numpy as np

class TransformImage():
    
    def __init__(self, ImageDataGenerator, 
                 dir_input='',
                 dir_output='',
                 resize_max_pixel=None,
                 prefix_rename='',
                 #is_autosave=False,                
                ):
        self.ImageDataGenerator = ImageDataGenerator
        self.dir_input = dir_input + '/'
        self.dir_output = dir_output + '/'
        self.resize_max_pixel = resize_max_pixel
        self.prefix_rename = prefix_rename
        #self.is_autosave = is_autosave
    
    def transform(self, filename):
        img01 = image.load_img(self.dir_input+filename)
        size = img01.size
        if self.resize_max_pixel is not None:
            ratio = size[0]*size[1]/self.resize_max_pixel
            new_size = (int(size[0]/ratio), int(size[1]/ratio))
            img01 = img01.resize(new_size)
            size = img01.size
        my_image = np.array(img01).reshape(1, size[1], size[0], 3)
        return image.array_to_img((next(self.ImageDataGenerator.flow(my_image))[0]))
    
    def transform_and_save(self, filename, specific_prefix=''):
        image = self.transform(filename)
        
        # Définition du nouveau nom
        all_prefix_name = str(uuid.uuid4()).split('-')[0]
        
        if specific_prefix!='':
            all_prefix_name = specific_prefix + '_' + all_prefix_name
        
        if self.prefix_rename!='':
            all_prefix_name = self.prefix_rename + '_' + all_prefix_name 
        
        new_name = all_prefix_name + '_' + filename
        image.save(self.dir_output + '/' + new_name)
        return new_name

# Generation de la duplication des Images avec Transformation
def genere_list_name_from_a_class(name, df, nb_to_create=5):
    '''Génération d'un DataFrame avec le nom des image à dupliquer'''
    l1 = list(df[df.Id==name].Image.values)*nb_to_create
    # On laisse jouer le random pour permuter les liste
    shuffle(l1)
    df_tmp = pd.DataFrame({'Image': l1[:nb_to_create]})
    df_tmp['Id'] = name
    return df_tmp

def create_df_with_image_to_augment(df_in, min_occurence=6):
    # On identifie l'occurrence de chaque classe
    frequences = df_in.Id.value_counts()
    
    # On ne traite que celles manquant d'image
    df_to_treat = frequences[frequences.values<min_occurence]
    
    l_df_with_names = []
    
    # On lance la génération d'image classe par classe
    for name, nb_occ in zip(df_to_treat.index, df_to_treat.values):
        tmp_df = genere_list_name_from_a_class(
            name, 
            df_in, 
            nb_to_create=min_occurence-nb_occ)
        l_df_with_names.append(tmp_df)
     
    # On renvoie le dataframe généré
    return pd.concat(l_df_with_names).reset_index(drop=True)

def genere_image_aug(df_in, myImageDataGenerator, 
        min_occurence=6,
        path_in='./../DATAS/data/train',
        path_out = './../DATAS/data/aug',
        ):
    '''
    Fonction qui 
    ------------
    - recupere la liste des images a dupliquer
    - genere les nouvelles images
    - renvoie un dataframe avec les nouveau nom et les Id

    Lancement:
    ----------
    - df_image_aug = genere_image_aug(df_train, train_datagen, 
            min_occurence=6, 
            path_in='./../DATAS/data/train',
            path_out = './../DATAS/data/aug',
            )
    '''
    
    # Liste des images à dupliquer
    new_dataframe = create_df_with_image_to_augment(df_in, min_occurence=6)
    
    # Les Paths
    
    

    # On Initialise la classe de tranformation des Images
    MyTransformation = TransformImage(
        train_datagen, 
        resize_max_pixel=300000, 
        dir_input=path_in, 
        dir_output=path_out, 
        prefix_rename='AUG')

    idx_nb_traite = 0
    l_new_name = []
    # On lance la transformation des Images
    for m_name, m_class in zip(new_dataframe.Image, new_dataframe.Id):
        idx_nb_traite+=1
        if (idx_nb_traite%1000==0):
            print('-- {} / {}'.format(idx_nb_traite, new_dataframe.shape[0]))
        
        new_name = MyTransformation.transform_and_save(filename=m_name)
        l_new_name.append(new_name)
        
    return pd.DataFrame({'Image': l_new_name, 'Id': new_dataframe.Id})    


