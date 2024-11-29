#For Genshin 5.0, August 2024
# 1. Import required libraries
import pandas as pd
import matplotlib.pyplot as plt
import random

from scipy.cluster.hierarchy import linkage, dendrogram, fcluster

from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import  confusion_matrix, classification_report

# 2. Load the data
data = pd.read_csv('genshin2.csv', delimiter = ';')
data = data[['character_name', 'DPS?', 'Sub DPS?', 'Support?']] 

# 3. Define features
characters = data['character_name']
DPS = data['DPS?']
SubDPS = data['Sub DPS?']
Support = data['Support?']

# 4. Implement team selection
team = characters.values.tolist()
random.shuffle(team)
team = team[0:4]
print(team)

# 5. Retrieve details about the random characters
def details():
    character_details = {}
    for character in team:
        character_row = data[data['character_name'] == character]
        if character_row['DPS?'].values[0] == 'Yes':
            character_details[character] = 'DPS'
        elif character_row['Sub DPS?'].values[0] == 'Yes':
            character_details[character] = 'Sub DPS'
        elif character_row['Support?'].values[0] == 'Yes':
            character_details[character] = 'Support'
        else:
            character_details[character] = 'Neutral'

    return character_details
# Call the function and store the output
character_details = details()
# Print the stored output (optional)
for character, role in character_details.items():
    print(f"{character} is a {role} character")
# 6. Evaluate team
def evaluate_team(character_details):
    dps_count = sum(1 for role in character_details.values() if role == 'DPS')
    sub_dps_count = sum(1 for role in character_details.values() if role == 'Sub DPS')
    support_count = sum(1 for role in character_details.values() if role == 'Support')
    if dps_count >= 1 and sub_dps_count >= 1 and support_count >= 1:
        print('Your team is an S tier team')
    elif dps_count >= 1 and sub_dps_count >= 1:
        print('Your team is an A tier team')
    elif dps_count >= 1 and support_count >= 1:
        print('Your team is an A tier team')
    elif dps_count >= 1:
        print('Your team is a B tier team')
    elif sub_dps_count >= 1 and support_count >= 1:
        print('Your team is a B tier team')
    elif sub_dps_count >= 1:
        print('Your team is a B tier team')
    else:
        print('Your team is a C tier team')
# Now call the evaluate_team function with the character_details
evaluate_team(character_details)

# 7. Add Hierarchical clustering
df = pd.get_dummies(data)
Z = linkage(df, method='ward')
dendrogram(Z, orientation='left', truncate_mode = 'level', p=2)
plt.show()
labels = fcluster(Z, t=3, criterion='distance')
df['cluster'] = labels
print(df)

# 8. Evaluate dendrogram
print("Supports are the most diverse group, while DPS has the most data in the group. SubDPS shares a lot of similarities with the DPS group, since it is included as a sub-group of the DPS clusters.")
