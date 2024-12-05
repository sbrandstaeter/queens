#
# SPDX-License-Identifier: LGPL-3.0-or-later
# Copyright (c) 2024, QUEENS contributors.
#
# This file is part of QUEENS.
#
# QUEENS is free software: you can redistribute it and/or modify it under the terms of the GNU
# Lesser General Public License as published by the Free Software Foundation, either version 3 of
# the License, or (at your option) any later version. QUEENS is distributed in the hope that it will
# be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
# FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more details. You
# should have received a copy of the GNU Lesser General Public License along with QUEENS. If not,
# see <https://www.gnu.org/licenses/>.
#
"""Utils to create codeclimate json from pylint."""

import hashlib
import json
import sys
from pathlib import Path


def main(pylint_json_path):
    """Convert pylint json to gitlab json.

    Inspired from
    https://gist.github.com/caryan/87bdadba4b6579ffed8a87d546364d72.
    """
    pylint_json_path = Path(pylint_json_path)
    pylint_json = json.loads(pylint_json_path.read_text(encoding="utf-8"))

    codeclimate_json = []
    for code_warning in pylint_json:
        code_climate_warning = {
            "description": code_warning["message"],
            "check_name": code_warning["symbol"],
            "severity": "minor",
            "fingerprint": hashlib.sha1(
                (code_warning["symbol"] + code_warning["message"] + code_warning["path"]).encode()
            ).hexdigest(),
            "location": {"path": code_warning["path"], "lines": {"begin": code_warning["line"]}},
        }
        codeclimate_json.append(code_climate_warning)

    Path("pylint_codeclimate.json").write_text(
        json.dumps(codeclimate_json, indent=4), encoding="utf-8"
    )


if __name__ == "__main__":
    main(sys.argv[1])
