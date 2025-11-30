import yaml

class FormattedConfig:
    algorithm = ""
    features = list()
    targets = list()

    def __init__(self,algorithm,features,targets):
        self.algorithm = algorithm
        self.features = features
        self.targets = targets

def parse_mode_config(config_path) -> FormattedConfig:
    d = dict()

    with open(config_path, 'r') as stream:
        try:
            d=yaml.safe_load(stream)
        except yaml.YAMLError as e:
            print(e)

    model_dict = d["Model"]

    algorithm = next(iter(model_dict))

    algorithm_dict = model_dict[algorithm]

    features = algorithm_dict["Features"]
    targets = algorithm_dict["Targets"]

    obj = FormattedConfig(algorithm=algorithm,features=features,targets=targets)

    return obj

    