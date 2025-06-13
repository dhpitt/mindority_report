#!/bin/bash

cd frontend; npm run dev; sleep 5;

cd ..; uv run -- dora start dataflow.yml --detach --name $1 