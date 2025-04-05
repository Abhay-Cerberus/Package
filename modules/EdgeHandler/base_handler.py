class BaseModelHandler:
    def optimize(self, techniques):
        raise NotImplementedError

    def convert(self, target_format):
        raise NotImplementedError

    def evaluate(self, test_data):
        raise NotImplementedError
