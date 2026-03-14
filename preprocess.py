import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
import joblib

df = pd.read_csv(r"D:\WA_Fn-UseC_-HR-Employee-Attrition.csv\WA_Fn-UseC_-HR-Employee-Attrition.csv")
df['Attrition'] = (df['Attrition'] == 'Yes').astype(int)

# Feature engineering
df['IncomeExperienceRatio'] = df['MonthlyIncome'] / (df['TotalWorkingYears'] + 1)
df['PromotionDelay'] = (df['YearsSinceLastPromotion'] > 3).astype(int)
df['EngagementScore'] = (df['EnvironmentSatisfaction'] + df['JobSatisfaction'] + df['RelationshipSatisfaction']) / 3
df['WorkloadStress'] = ((df['OverTime'] == 'Yes') & (df['JobInvolvement'] > 2)).astype(int)

cat_cols = ['BusinessTravel', 'Department', 'EducationField', 'Gender', 'JobRole', 'MaritalStatus', 'OverTime']
num_cols = ['Age', 'DailyRate', 'DistanceFromHome', 'Education', 'HourlyRate', 'JobLevel', 
            'MonthlyIncome', 'NumCompaniesWorked', 'PercentSalaryHike', 'PerformanceRating',
            'StockOptionLevel', 'TotalWorkingYears', 'TrainingTimesLastYear', 'WorkLifeBalance',
            'YearsAtCompany', 'YearsInCurrentRole', 'YearsWithCurrManager', 'IncomeExperienceRatio',
            'PromotionDelay', 'EngagementScore', 'WorkloadStress']

X = df.drop('Attrition', axis=1)[cat_cols + num_cols]
y = df['Attrition']

le_dict = {}
for col in cat_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    le_dict[col] = le

preprocessor = ColumnTransformer([
    ('num', StandardScaler(), num_cols),
    ('cat', 'passthrough', cat_cols)
])

X_pre = preprocessor.fit_transform(X)
smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X_pre, y)

X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, stratify=y_res, random_state=42)

joblib.dump(le_dict, 'label_encoders.pkl')
joblib.dump(preprocessor, 'preprocessor.pkl')
joblib.dump(X_train, 'X_train.pkl')
joblib.dump(y_train, 'y_train.pkl')
