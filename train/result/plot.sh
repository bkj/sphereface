#!/bin/bash
cat $1 | fgrep -a " loss =" | awk -F'=' '{print $2}' | python ./plot.py 
