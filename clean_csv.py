import csv
import pandas as pd
# ds = pd.read_csv('./heart_failure.csv')
# ds = pd.read_csv('./heart.csv')

Index = ['Age', 'RestingBP', 'Cholesterol', 'FastingBS', 'MaxHR', 'Oldpeak','HeartDisease', 'ChestPainType = ATA', 'ChestPainType = NAP','ChestPainType = ASY', 'RestingECG = Normal', 'RestingECG = ST','RestingECG = LVH', 'ExerciseAngina = N', 'ExerciseAngina = Y','ST_Slope = Up', 'ST_Slope = Flat', 'Sex = M', 'Sex = F']
data = []
with open('heart_failure.csv', 'r') as file:
    csvreader = csv.reader(file)
    for row in csvreader:
        if row[2] == '0.0':
            continue
        else:
            data.append(row)
        
    file.close()

df = pd.DataFrame(data[1:], columns=data[0])
df.to_csv('heart_failure2.csv', index=False)

