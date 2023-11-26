class HParams:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def __str__(self):
        return str(self.__dict__)
    
    def override_from_dict(self, override_dict):
        for key, value in override_dict.items():
            if hasattr(self, key):
                setattr(self, key, value)