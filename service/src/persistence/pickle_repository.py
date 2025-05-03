import joblib
from . import ModelRepository
MODELS_CACHE_DIR='./cache'
DEFAULT_NAME = ""

class PickleRepository(ModelRepository):
    def common_name(self, hash_model, name):
        return f'{name}-{hash_model}.pkl'
    def load(self, name=DEFAULT_NAME):
        hash_id = 'prod'
        return joblib.load(f'{MODELS_CACHE_DIR}/{self.common_name(hash_id, name)}')
    def save(self, model, name=DEFAULT_NAME):
        hash_id = 'prod'
        print(f'Saving: {self.common_name(hash_id, name)}')
        joblib.dump(model, f'{MODELS_CACHE_DIR}/{self.common_name(hash_id, name)}')
