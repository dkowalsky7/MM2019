import pandas as pd
import numpy as np
import os
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import log_loss
np.random.seed(505)


def formatEntry(compName,year=2019):
    data = pd.read_csv(os.path.join(compName, 'SampleSubmissionStage2.csv'))
    eline = data.ID.str.split(pat='_', expand=True).apply(pd.to_numeric)
    eline.columns = ['Season', 'LTeam', 'RTeam']
    output = pd.concat([data, eline], sort=False, axis=1)
    return output[output.Season == year].reset_index(drop=True)
    # return output
def formatEstimate(compName,year=2018,minYear=2001):
    data = pd.read_csv(os.path.join(compName, 'DataFiles/NCAATourneyCompactResults.csv'))
    data['Pred'] = (data.WTeamID < data.LTeamID).astype(int)
    data = data[data.Season.between(minYear,year-1)]
    formed = pd.DataFrame(np.sort(data[['WTeamID','LTeamID']],axis=1),
                          columns=['LTeam','RTeam'])

    formed['Pred'] = data.Pred.values
    formed['Season'] = data.Season.values
    formed['ID'] = formed.Season.astype(str) + '_' + \
                   formed.LTeam.astype(str) + '_' + \
                   formed.RTeam.astype(str)
    return formed

def SeedMe(compName,data):
    seedFile = pd.read_csv(os.path.join(compName,'DataFiles/NCAATourneySeeds.csv'))
    data = data.merge(right=seedFile,left_on=['Season','LTeam'],
                      right_on=['Season','TeamID'],how='left').drop(
        'TeamID',axis=1).rename(
        columns={'Seed':'SeedL'})

    data = data.merge(right=seedFile, left_on=['Season', 'RTeam'],
                      right_on=['Season', 'TeamID'], how='left').drop(
        'TeamID', axis=1).rename(
        columns={'Seed': 'SeedR'})
    data['RegL'] = data.SeedL.str.extract('([a-zA-Z]+)',expand=False)
    data['RegR'] = data.SeedR.str.extract('([a-zA-Z]+)', expand=False)
    data['SeedL'] = data.SeedL.str.extract('(\d+)', expand=False).astype(int)
    data['SeedR'] = data.SeedR.str.extract('(\d+)', expand=False).astype(int)

    data['Upset'] = ((data.SeedL >= data.SeedR) & (data.Pred == 1)) |\
                    ((data.SeedR >= data.SeedL) & (data.Pred == 0))
    data.Upset = data.Upset.astype(int)

    data['SeedL'] = data.SeedL // 2
    data['SeedR'] = data.SeedR // 2


    dummies = pd.get_dummies((data.SeedL-data.SeedR).abs(),prefix='SDiff')

    return pd.concat([data,dummies],axis=1)

def Record(compName,data):
    recFile = pd.read_csv(os.path.join(compName,'DataFiles/RegularSeasonCompactResults.csv'))
    wins = recFile.groupby(['Season','WTeamID']).size().reset_index().rename(columns={0:'Wins','WTeamID':'TeamID'})
    loses = recFile.groupby(['Season','LTeamID']).size().reset_index().rename(columns={0:'Loses','LTeamID':'TeamID'})
    combo = pd.merge(wins,loses,on=['Season','TeamID'],how='outer').fillna(0)
    data = data.merge(combo,left_on=['Season','LTeam'],right_on=['Season','TeamID'],how='left').drop(
        'TeamID',axis=1).rename(columns={'Wins':'LWins','Loses':'LLoses'})

    data = data.merge(combo, left_on=['Season', 'RTeam'], right_on=['Season', 'TeamID'], how='left').drop(
        'TeamID', axis=1).rename(columns={'Wins': 'RWins', 'Loses': 'RLoses'})

    data['LWPC'] = data['LWins'] / (data['LWins'] + data['LLoses'])
    data['RWPC'] = data['RWins'] / (data['RWins'] + data['RLoses'])
    data['DiffWPC'] = data['LWPC'] - data['RWPC']
    return data

def DivTourny(compName,data):
    conFre = pd.read_csv(os.path.join(compName,'DataFiles/ConferenceTourneyGames.csv'))
    conFre = conFre.groupby(['Season','WTeamID']).size().reset_index().rename(columns={0:'ConDist',
                                                                                    'WTeamID':'TeamID'})

    con = pd.merge(data,conFre,left_on=['Season','LTeam'],right_on=['Season','TeamID'],how='left').drop(
        'TeamID',axis=1).rename(columns={'ConDist':'LConDist'})
    con = pd.merge(con, conFre, left_on=['Season', 'RTeam'], right_on=['Season', 'TeamID'], how='left').drop(
        'TeamID', axis=1).rename(columns={'ConDist': 'RConDist'})

    con['LConDist'].fillna(0,inplace=True)
    con['RConDist'].fillna(0,inplace=True)
    dummys = pd.get_dummies((con.LConDist.astype(int) - con.RConDist.astype(int)).abs(),prefix='ConDist')

    return pd.concat([con,dummys],axis=1)

def addFactors(dataFramework,compName):
    #Add seed information
    dataFramework = SeedMe(Tourney, dataFramework)
    #Add Season Record
    dataFramework = Record(compName,dataFramework)
    #Add Conference
    dataFramework = DivTourny(compName,dataFramework)

    return dataFramework

def Model(data,estimate='Upset'):
    test = data[data.Season == 2018].copy()
    train = data[data.Season < 2018]

    x = [f"SDiff_{z}" for z in range(1,9)] + [f"ConDist_{r}" for r in range(0,5)] + ['DiffWPC']
    logit = LogisticRegression()
    logit.fit(train[x],train[estimate])
    print("Accuracy = ",logit.score(test[x],test[estimate]))
    print('LogLoss  = ',log_loss(test[estimate],logit.predict_proba(test[x])[:,1]))

    print(logit.predict_proba(test[x])[:,1])

    test['Acutals'] = logit.predict_proba(test[x])[:,1]
    fixPreds(test)

    # print(test)


def fixPreds(data):
    data.to_csv('Check.csv')

if __name__ == '__main__':
    Tourney = 'MensComp'
    entryData = formatEntry(Tourney,year=2019)
    predicationData = formatEstimate(Tourney,year=2019)
    setter = pd.concat([entryData,predicationData],axis=0,sort=False)
    output = addFactors(setter,Tourney)

    Model(output,estimate='Upset')

    # print(output)
    # output.to_csv('TestOut.csv')
