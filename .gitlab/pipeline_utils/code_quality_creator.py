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
            "location": {"path": code_warning["path"], "lines": {"begins": code_warning["line"]}},
        }
        codeclimate_json.append(code_climate_warning)

    Path("pylint_codeclimate.json").write_text(
        json.dumps(codeclimate_json, indent=4), encoding="utf-8"
    )


if __name__ == "__main__":
    main(sys.argv[1])
