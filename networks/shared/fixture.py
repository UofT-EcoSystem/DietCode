import abc


class ModelFixture(abc.ABC):
    __slots__ = 'model', 'scripted_model', 'input_data_torch', 'input_data_np'

    @property
    @abc.abstractmethod
    def name(self):
        pass

    @property
    @abc.abstractmethod
    def input_name(self):
        pass
