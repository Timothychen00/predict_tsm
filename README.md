# predict_tsm
## Environment Setup
I use "pipenv" to manage all the requirements used in this project.
use below commands you will be sure that your environment is all up to date and ready to power the whole project.
```python
pip install pipenv
pipenv sync
```

## Run 
### Setup Backend
```python
    pipenv run python server.py
```
### Setup Frontend 
```python
    pipenv run python -m frontend.app
```

### Use Backend Without Using Web view
```python
    pipenv run python -m backend.model
```
