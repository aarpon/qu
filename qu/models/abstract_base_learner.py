#   /********************************************************************************
#   * Copyright © 2020-2021, ETH Zurich, D-BSSE, Aaron Ponti
#   * All rights reserved. This program and the accompanying materials
#   * are made available under the terms of the Apache License Version 2.0
#   * which accompanies this distribution, and is available at
#   * https://www.apache.org/licenses/LICENSE-2.0.txt
#   *
#   * Contributors:
#   *     Aaron Ponti - initial API and implementation
#   *******************************************************************************/
#

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
