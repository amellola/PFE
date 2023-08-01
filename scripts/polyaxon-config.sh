POLYAXON_USERNAME=$(awk '{print $1}' polyaxon/pass.txt)
POLYAXON_PASSWORD=$(awk '{print $2}' polyaxon/pass.txt)

POLYAXON_HOST=$(awk '$1=="host" {print $2}' polyaxon/config.cfg)
POLYAXON_PORT=$(awk '$1=="port" {print $2}' polyaxon/config.cfg)

polyaxon config set --host=$POLYAXON_HOST --port $POLYAXON_PORT
polyaxon login --username $POLYAXON_USERNAME --password $POLYAXON_PASSWORD
