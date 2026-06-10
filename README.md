# HELICS-Examples

[![Build Status](https://dev.azure.com/HELICS-test/HELICS-Examples/_apis/build/status/GMLC-TDC.HELICS-Examples?branchName=master)](https://dev.azure.com/HELICS-test/HELICS-Examples/_build/latest?definitionId=2?branchName=master)

Examples for using HELICS with a variety of supported programming languages.

All C and C++ examples can be built with CMake. They can be built from
individual folders or by running CMake from the repository root.

On Windows and systems with HELICS installed in a non-system search path, set
the `HELICS_DIR` environment variable to the HELICS install folder, which
contains the `bin`, `include`, and `lib`/`lib64` subfolders. Alternatives
include adding the HELICS install folder to `PATH` or setting
`CMAKE_PREFIX_PATH`, either as an environment variable or with the
`-DCMAKE_PREFIX_PATH=value` CMake argument.

## Source Repo

The HELICS source code is hosted on GitHub: [https://github.com/GMLC-TDC/HELICS](https://github.com/GMLC-TDC/HELICS)

## Citation
T. Hardy, B. Palmintier, P. Top, D. Krishnamurthy and J. Fuller, "HELICS: A Co-Simulation Framework for Scalable Multi-Domain Modeling and Analysis," in IEEE Access, doi: 10.1109/ACCESS.2024.3363615, available at [https://ieeexplore.ieee.org/document/10424422](https://ieeexplore.ieee.org/document/10424422)

## Release
HELICS-Examples is distributed under the terms of the BSD-3-Clause license. All
new contributions must be made under this license. See [LICENSE](LICENSE).

SPDX-License-Identifier: BSD-3-Clause

Portions of the code were written by LLNL with release number
LLNL-CODE-739319
