#/bin/bash

echo 'Running isort.'
isort -rc ./scripts

echo 'Running black.'
black ./scripts

echo 'Finished auto formatting.'
