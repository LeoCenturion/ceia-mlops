import joblib
import os

MODELS_CACHE_DIR='./cache'
DEFAULT_NAME = ""

def hash_model(model):
    return 'prod'

def common_name(hash_model, name):
    return f'{name}-{hash_model}.pkl'

def load(model, name=DEFAULT_NAME):
    hash_id = hash_model(model)
    return joblib.load(f'{MODELS_CACHE_DIR}/{common_name(hash_id, name)}')

def save(model, name=DEFAULT_NAME):
    hash_id = hash_model(model)
    print(f'Saving: {common_name(hash_id, name)}')
    joblib.dump(model, f'{MODELS_CACHE_DIR}/{common_name(hash_id, name)}')

def clean(model, name=DEFAULT_NAME):
    hash_id = hash_model(model)
    os.remove(f'{MODELS_CACHE_DIR}/{common_name(hash_id, name)}')

def load_or_fit(model, x, y, name = DEFAULT_NAME, force = False):
    fitted = None
    try:
        if force: clean(model, name)
        fitted = load(model, name)
    except FileNotFoundError:
        fitted = model.fit(x,y)
        save(fitted, name)
    return fitted
