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

from enum import IntEnum


class Architectures(IntEnum):
    ResidualUNet2D = 0,
    AttentionUNet2D = 1


class Losses(IntEnum):
    GeneralizedDiceLoss = 0


class Optimizers(IntEnum):
    Adam = 0,
    SGD = 1
