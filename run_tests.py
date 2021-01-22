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

import unittest
import tests

# Run all unit tests
suite = unittest.TestLoader().loadTestsFromModule(tests)
result = unittest.TextTestRunner().run(suite)

print(f"Run {result.testsRun} tests: "
      f"{len(result.errors)} error(s) and "
      f"{len(result.failures)} failure(s).")
