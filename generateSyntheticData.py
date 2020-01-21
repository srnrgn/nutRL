#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  9 10:11:12 2019

@author: Sema
From a dataframe about user information, create a profile for a user. 

For generating data

"""
import logging
import pickle
import random
import numpy as np
import pandas as pd
from itertools import permutations
from userModel import User, Food, Meal, UserDiet, Substitution
from policies import System

#logging.basicConfig(level=logging.DEBUG)


def createUserObject(df):
    """
    From a dataframe, create User objects.
    
    Returns a dict of User objects
    """
    U = {}
    for index, row in df.iterrows():
        user = User(row.nomen, row.sexe_ps, row.v2_age, row.poidsd, row.poidsm,
                    row.bmi,row.menopause, row.regmaig, row.regmedic, 
                    row.regrelig, row.regvegr, row.regvegt, row.ordi, 
                    row.bonalim, row.agglo9)
        U[row.nomen] = user
    return U


def eraseDubloons(foodList):
    """
    If the same item appears in the list, sums the quantities and create a 
    unique Food object.
    """
    resList = []
    foodNames = [f.codsougr_name for f in foodList]
    uniqueFoodNames = list(set(foodNames))
    for name in uniqueFoodNames:
        duplicates =  [f for f in foodList if f.codsougr_name == name]
        qte_nette = sum([f.qte_nette for f in duplicates])
        codsougr = [f.codsougr for f in duplicates][0]
        
        newFood = Food(codsougr, name, qte_nette)
        resList.append(newFood)
    return resList
        

def createConsumptionSeq(df):
    """
    For a single user, read from a dataframe the consumptions.
    """
    mealList = []
    M = {} #Contient les aliments d'un repas    
    nomen = df.nomen.iloc[0]
    jour = df.jour.iloc[0]
    tyrep = df.tyrep.iloc[0]
    
    for index, row in df.iterrows(): # Scan tous les aliments de consommation
        #print(row.nomen, row.jour, row.tyrep)
        if (row.nomen == nomen) & (row.jour == jour) & (row.tyrep == tyrep):
            name = str(row.nomen) + '_' + str(index)
            M[name] = Food(row.codsougr, row.codsougr_name, row.qte_nette) # Crée l'objet FoodItem
        else:
            m = list(set(M.values())) #Set of food items in a meal
            uniqueM = eraseDubloons(m)
            meal = Meal(uniqueM, jour = jour, tyrep = tyrep)
            meal.reorder()
            logging.debug('{}'.format(meal.nameTuple))
            if meal:
                mealList.append(meal)

            M = {}
            nomen = row.nomen
            jour = row.jour
            tyrep = row.tyrep
            
    meals = [m for m in mealList if m]
    return meals
    

def createMeal(jour, tyrep, itemsName, itemsDict, minItem, maxItem, meanQuantities):
    """
    Create a meal by selecting randomly items.
    """
    nItems = random.randint(minItem,maxItem)
    items = random.sample(itemsName, nItems)
    foodItems = [Food(itemsDict[x], x, meanQuantities[x]) for x in items]
    meal = Meal(foodItems, jour=jour, tyrep=tyrep)
    meal.reorder()
    return meal

def createDiet(itemsName, itemsDict, minItem, maxItem, nMeal, meanQuantities):
    """
    Create nMeal meals. 
    """
    diet = []
    for i in range(nMeal):
        tyrep = 1+i%3
        jour = 1 + i//3
        meal = createMeal(jour, tyrep, itemsName, itemsDict, minItem, maxItem, meanQuantities)
        diet.append(meal)
    return diet

def generateAcceptabilityMatrix(meals, items):
    """
    Create a dataframe with meals in row and tuple of items in columns. 
    Represents the acceptability of a substitution given a meal.
    """
    meals_ = [m for m in meals if m]
    substitutions = list(permutations(items,2))
    nMeals = len(meals_)
    nSub = len(substitutions)
    A = pd.DataFrame(np.random.uniform(0,1, (nMeals, nSub)), index = meals_, columns = substitutions)
    logging.debug('Created acceptability matrix randomly')
    return A

def generateNoisyMatrix(A, mu, sigma):
    """
    Generate a matrix with Gaussian noise.
    
    The result matrix must be positive
    """
    noise = np.random.normal(mu, sigma, A.shape)
    noisyMat = A + noise
    # noisyMat values must be in [0,1]
    noisyMat[noisyMat <= 0] = A.min().min()
    noisyMat[noisyMat > 1] = 1 
    logging.debug('Created noisy matrix with sigma noise {}'.format(sigma))
    return noisyMat


def createUserDietObject(user, itemsName, itemsDict, minItem, maxItem, nMeal, 
                         portions, composition, socio, adequacyRef, moderationRef, 
                         nLastDays):
    """
    Input :
        user (UserObject)
        itemsNames (list of strings) names of food items
        itemsDict (dict) name:id
        minItem (int) minimum meal length
        maxItem (int) maximum meal length
        nMeal (int) number of meal to generate
        meanQuantities (dict) itemId:qty --> mean quty consumed for given item
        
    """
    nbItem = len(itemsName)
    
    if user.sexe_ps == 1:
        meanQuantities = portions
    
    diet = createDiet(itemsName, itemsDict, minItem, maxItem, nMeal, meanQuantities)
    meals = [x.nameTuple for x in diet if x.nameTuple]
    ms = list(set(meals))
    A = generateAcceptabilityMatrix(ms, itemsName)
    
    userDiet = UserDiet(user, diet, nbItem, A, meanQuantities, itemsDict, 
                portions, composition, socio, adequacyRef, moderationRef, 
                nLastDays)
    return userDiet 
    
    
#### Import data 
#conso = pd.read_csv('data/testConso.txt')
#adequacyRef = pd.read_csv('data/adequacyReferences.csv', index_col=[0,1], header=0,
#                          delimiter=";")
#moderationRef = pd.read_csv('data/moderationReferences.csv', index_col=[0,1], header=0,
#                          delimiter=";")
#
#with open('data/compoCodsougr.p', 'rb') as handle:
#    composition = pickle.load(handle) 
#
#with open('data/meanQuantitiesDict.p', 'rb') as handle:
#    meanQuantities = pickle.load(handle) 
#    
#with open('data/dictCodsougr.p', 'rb') as handle:
#    dict_codsougr = pickle.load(handle) 
#
#with open('data/portions.p', 'rb') as handle:
#    portions = pickle.load(handle) 
#
#portions = {i+1:v for i,v in enumerate(portions)}
#
#
#itemsDict = {v:k for k,v in dict_codsougr.items()}
#itemsName = list(dict_codsougr.values())
##itemsName = ['bread', 'coffee', 'yoghurt', 'tea infusion', 'butter', 
##             'jam honey', 'rice', 'beef', 'spring water', 'beer']
#
#

