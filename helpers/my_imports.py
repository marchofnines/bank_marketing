import sys
sys.path.append('/Users/basilhaddad/jupyter/module17/bank_marketing_repo/')
import pandas as pd
import numpy as np
from IPython.display import display
from IPython.core.display import HTML
from pandas.io.formats.style import Styler

import matplotlib.pyplot as plt
import seaborn as sns
import scikitplot as skplt
import plotly.express as px
import plotly.subplots as sp
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.io as pio
pio.renderers.default='notebook'

from scipy.stats import entropy, randint, uniform
import scipy.stats as stats
from scipy.linalg import svd

from sklearn.experimental import enable_halving_search_cv, enable_iterative_imputer
from sklearn.impute import SimpleImputer, IterativeImputer, KNNImputer
from category_encoders import JamesSteinEncoder, BinaryEncoder
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler, MaxAbsScaler, QuantileTransformer
from sklearn.preprocessing import PolynomialFeatures, OneHotEncoder, OrdinalEncoder, LabelBinarizer
from sklearn.compose import make_column_transformer, make_column_selector, ColumnTransformer, TransformedTargetRegressor
from sklearn.feature_selection import SequentialFeatureSelector, SelectFromModel, RFE
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN

from sklearn.linear_model import LinearRegression, Lasso, HuberRegressor, Ridge, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.multiclass import OneVsOneClassifier
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier, export_text, plot_tree
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score , train_test_split
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, HalvingGridSearchCV, HalvingRandomSearchCV

from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_squared_error, accuracy_score, f1_score, recall_score, precision_score, make_scorer
from sklearn.metrics import precision_recall_curve, auc, roc_curve, RocCurveDisplay, log_loss, roc_auc_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from sklearn.inspection import PartialDependenceDisplay, partial_dependence

from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE 
from imblearn.under_sampling import RandomUnderSampler
from collections import Counter

from sklearn import set_config
set_config(display="diagram")
#set_config("figure")
import warnings
warnings.filterwarnings('ignore')


from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)
