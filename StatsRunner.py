import pandas as pd
import numpy as np
import os
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import log_loss,accuracy_score
from sklearn.model_selection import train_test_split

def SeedMe(df,seeds,rounds):
    """
    Prep model Feeding Data, Add Seeds, Add the Round the Tourney is being played.
    :param df: Tourney Compact Results,
    :param seeds: File of Current Seeds
    :param rounds: Custom File, with round numbers
    :return:
    """
    seeds['region'] = seeds.Seed.str[:1]
    seeds.Seed = seeds.Seed.str.extract('(\d+)', expand=False).astype(int)

    df = df.merge(seeds,left_on=['Season','WTeamID'],right_on=['Season','TeamID'],how='left').drop(['TeamID'],axis=1).merge(
        seeds,left_on=['Season','LTeamID'],right_on=['Season','TeamID'],how='left',suffixes=('W','L')
    ).drop(['TeamID'],axis=1)

    sr = np.sort(df.loc[df.regionW == df.regionL,['SeedW','SeedL']].values,axis=1)

    sr = pd.DataFrame(sr,columns=['S1','S2'])
    rounds.iloc[:,1:] = np.sort(rounds.values[:,1:],axis=1)

    sr = sr.merge(rounds,on=['S1','S2'],how='left')

    df.loc[df.regionW == df.regionL,'round'] = sr['round'].values
    l1 = ['W','X']
    l2 = ['Y','Z']

    df.loc[(df.regionW != df.regionL) & (df.regionW.isin(l1)) & (df.regionL.isin(l1)),'round'] = 5
    df.loc[(df.regionW != df.regionL) & (df.regionW.isin(l2)) & (df.regionL.isin(l2)),'round'] = 5
    df.loc[(df.SeedW == df.SeedL) & (df.regionW == df.regionL),'round'] = 0
    df['round'] = df['round'].fillna(6)

    df['Y'] = (df.SeedW >= df.SeedL).astype(int)

    return df

def contournyR(dfIn,ConferenceDf):
    gb = ConferenceDf.groupby(['Season','WTeamID'],as_index=False).DayNum.count().rename(columns={'WTeamID':'TeamID','DayNum':'Tdist'})
    x = dfIn.merge(gb.rename(columns={'TeamID': 'WTeamID', 'Tdist': 'WTdist'}), on=['Season', 'WTeamID'],how='left') \
        .merge(gb.rename(columns={'TeamID': 'LTeamID', 'Tdist': 'LTdist'}), on=['Season', 'LTeamID'],how='left')
    return x.fillna(0)

def Logit(Data):
    X_train,X_test,Y_train,Y_test = train_test_split(Data[['WTdist','LTdist']].values, Data['Y'].values)
    logisticRegr = LogisticRegression(solver='lbfgs', fit_intercept=True)
    logisticRegr.fit(X_train,Y_train)
    print(logisticRegr.score(X_test,Y_test))



MFiles = 'mens-machine-learning-competition-2019/DataFiles'

TComp = pd.read_csv(os.path.join(MFiles,'NCAATourneyCompactResults.csv'))
TComp = TComp[TComp.Season >= 2001]
seedsFiles = pd.read_csv(os.path.join(MFiles,'NCAATourneySeeds.csv'))
Trounds = pd.read_csv(os.path.join(MFiles,'Rounds.csv'))
conFerenceTourny = pd.read_csv(os.path.join(MFiles,'ConferenceTourneyGames.csv'))
output = SeedMe(TComp,seedsFiles,Trounds)

output = contournyR(output,conFerenceTourny)
Logit(output)

# output.to_csv('Data.csv')
# print(output)
