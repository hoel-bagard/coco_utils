### Requirements generation

The requirements can be re-generated with the following commands
```
pip-compile --output-file=requirements/requirements.txt --resolver=backtracking pyproject.toml
pip-compile --extra=test --output-file=requirements/requirements-test.txt --resolver=backtracking pyproject.toml
pip-compile --extra=dev --output-file=requirements/requirements-dev.txt --resolver=backtracking pyproject.toml
```
