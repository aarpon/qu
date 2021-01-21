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


class ImportTestCase(unittest.TestCase):

    def test_import(self):
        failed = False
        try:
            import qu
        except Exception as e:
            failed = True
            print(e)

        self.assertEqual(False, failed)


if __name__ == '__main__':
    unittest.main()
