# Data pre-processing and handling libraries
import numpy as np
import pandas as pd

# Visualization libraries
import matplotlib.pyplot as plt
import seaborn as sns

# Machine learning realted imports
from sklearn.model_selection import train_test_split
from scipy.stats import ttest_ind, chi2_contingency
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import f1_score, roc_curve, roc_auc_score, precision_recall_curve, accuracy_score, classification_report
from sklearn.model_selection import cross_validate, LearningCurveDisplay, GridSearchCV, cross_val_predict, RandomizedSearchCV
from sklearn.preprocessing import PolynomialFeatures, StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import BaggingClassifier
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Extra libraries
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('./heart.csv')

print(df.head())