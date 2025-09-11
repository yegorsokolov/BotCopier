class TabTransformer:
    def __init__(self, *args, **kwargs):
        pass
    def load_state_dict(self, *args, **kwargs):
        pass
    def eval(self):
        pass
    def __call__(self, x):
        return x

def detect_resources():
    return {"lite_mode": True}
