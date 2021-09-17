from loader import BD_directory
from loader import intents_file
import os
import json
import pandas as pd

tables: dict = {}


def load_data():
    global tables
    for table in os.listdir(BD_directory):
        if table.split('.')[1] == 'csv':
            tables[table.split('.')[0]] = pd.read_csv(BD_directory + table,
                                                      encoding='utf-8', escapechar='\\', error_bad_lines=False)
    json_data = json.loads(open(intents_file).read())

    return json_data, tables


def get_specialty_td():
    example_specialty: pd.DataFrame = tables["hackathon_order_fix2"][['comment', 'specialty_id']].copy()
    example_specialty.dropna(inplace=True)
    specialty = tables['specialty'][['id', 'name']].copy()

    example_specialty = example_specialty.merge(specialty, how='left', left_on='specialty_id', right_on='id', copy=True)
    x_train = example_specialty['comment'].tolist()
    y_train = example_specialty['name'].tolist()
    return x_train, y_train


def get_disease_td():
    symptom: pd.DataFrame = tables['symptom'][['id', 'name']].copy()
    disease: pd.DataFrame = tables['disease'][['id', 'name']].copy()
    disease_symptom: pd.DataFrame = tables['disease_symptom'].copy()

    disease_symptom = disease_symptom.merge(symptom, how='left', left_on='symptom_id', right_on='id', copy=True)
    disease_symptom = disease_symptom.merge(disease, how='left', left_on='disease_id', right_on='id', copy=True)
    x_train = disease_symptom['name_x'].tolist()
    y_train = disease_symptom['name_y'].tolist()
    return x_train, y_train


def get_json_td(json_data):
    # slow
    x = []
    y = []
    for intent, intent_data in json_data['intents'].items():
        for example in intent_data['examples']:
            y.append(intent)
            x.append(example)
    return x, y


if __name__ == '__main__':
    load_data()
