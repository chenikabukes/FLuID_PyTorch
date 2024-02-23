#!/bin/bash
while IFS=' ' read -r ip rest
do 
    echo "$ip"
done <File.txt