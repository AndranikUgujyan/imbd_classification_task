class Literals:
    __INSTANCE = None

    def __init__(self):
        if Literals.__INSTANCE is None:
            Literals.__INSTANCE = self

    @staticmethod
    def get_instance():
        if Literals.__INSTANCE is None:
            Literals.__INSTANCE = Literals()
            return Literals.__INSTANCE

    @property
    def LOGGER_NAME(self):
        return 'sentiment_model'
