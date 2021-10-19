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

import re
import sys
from napari.__main__ import main
if __name__ == '__main__':
    sys.argv[0] = re.sub(r'(-script\.pyw|\.exe)?$', '', sys.argv[0])
    sys.exit(main())
