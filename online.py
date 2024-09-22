import numpy as np
from flask import Flask, template_rendered
import pandas as pd
import joblib
app = Flask(__name__)

filename = ''
model = joblib.load(open(filename, 'rb'))


