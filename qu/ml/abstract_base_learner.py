from abc import ABC, abstractmethod
from pathlib import Path
from typing import Union


class AbstractBaseLearner(ABC):

    @abstractmethod
    def train(self):
        raise NotImplementedError("Implement me!")

    @abstractmethod
    def predict(self,
                input_folder: Union[Path, str],
                target_folder: Union[Path, str],
                model_path: Union[Path, str]
                ):
        raise NotImplementedError("Implement me!")

    @abstractmethod
    def test_predict(
            self,
            target_folder: Union[Path, str] = '',
            model_path: Union[Path, str] = ''
    ) -> bool:
        raise NotImplementedError("Implement me!")
