pip install pipenv
pipenv sync
pipenv run python server.py & pipenv run python -m frontend.app &
# _pid=$!
# echo "$_pid" 

# #!/bin/bash
# function finish {
#   kill -9 "$_pid"
#   echo "1232312323232323"
# }
# trap finish EXIT