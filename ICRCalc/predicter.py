import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn import linear_model
import joblib

name = input("Name: ")
AC = int(input("AC: "))
HP = int(input("HP: "))
Str = int(input("Strength: "))
Dext = int(input("Dexterity: "))
Con = int(input("Constitution: "))
Int = int(input("Intelligence: "))
Wis = int(input("Wisdom: "))
Cha = int(input("Charisma: "))
Sav = int(input("# of saving throws: "))
Ski = int(input("# of skill proficiencies: "))
Dvu = int(input("# of damage vulnerabilities: "))
Dre = int(input("# of damage resistances: "))
Dim = int(input("# of damage immunities: "))
Cim = int(input("# of condition immunities: "))
Size = input("Size: ")
Overall = Str + Dext + Con + Int + Wis + Cha + Sav
monster = {"AC": AC, "HP": HP, "Saving Throws": Sav, "Skills": Ski, "Damage Vulnerabilities": Dvu,
           "Damage Resistances": Dre,
           "Damage Immunities": Dim, "Condition Immunities": Cim, "Size": Size, "Overall": Overall}

df_monster = pd.DataFrame.from_dict(monster, orient="index").T
dataset = pd.read_csv("prepared_dataset.csv")

from sklearn.feature_extraction.text import CountVectorizer

count_vect = CountVectorizer()
count_matrix = count_vect.fit_transform(df_monster["Size"])

df_monster = pd.concat(
    [df_monster, pd.DataFrame(data=count_matrix.toarray(), columns=count_vect.get_feature_names_out())], axis=1)
df_monster = df_monster.drop(columns=["Size"])

merged = pd.concat([dataset, df_monster], ignore_index=True)
merged = merged.fillna(0)
del merged["Name"]
del merged['Charisma']
del merged['Dexterity']
del merged['Intelligence']
del merged['Strength']
del merged['Wisdom']
del merged['Constitution']
del merged['CR']


columns_to_scale = ["HP"]
scaler = MinMaxScaler()
scaler.fit(merged[columns_to_scale])
merged[columns_to_scale] = scaler.transform(merged[columns_to_scale])


row = merged.iloc[-1:]
#with pd.option_context('display.max_rows', None, 'display.max_columns', None):
print(row["HP"])

lasso = joblib.load("model.sav")
print(lasso.predict(row))