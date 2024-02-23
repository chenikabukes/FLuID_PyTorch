#!/bin/bash

echo $1
cid=$(($1-1))
cid=$(($cid * 12))
echo $(($cid  + 1))
if [[ $1 -eq 0 ]]
then 
    echo $1
    hostname -I > ip.txt
    server="127.0.0.1"
    while IFS=' ' read -r ip rest
    do 
        echo "$ip"
        server="$ip"
    done <ip.txt
    python3 server.py --server_address $server:43101 --rounds 1000 --min_num_clients 100 --min_sample_size 1 --model Net

else 
    sleep 10
    echo $1
    server="127.0.0.1"
    while IFS=' ' read -r ip rest
    do 
        echo "$ip"
        server="$ip"
    done <ip.txt
    if [[ $cid -eq 84 ]]
    then 
        python3 client.py --server_address $server:43101 --cid $(($cid  + 0)) --model Net --device cpu &
        python3 client.py --server_address $server:43101 --cid $(($cid  + 1)) --model Net --device cpu &
        python3 client.py --server_address $server:43101 --cid $(($cid  + 2)) --model Net --device cpu &
        python3 client.py --server_address $server:43101 --cid $(($cid  + 3)) --model Net --device cpu &
        python3 client.py --server_address $server:43101 --cid $(($cid  + 4)) --model Net --device cpu &
        python3 client.py --server_address $server:43101 --cid $(($cid  + 5)) --model Net --device cpu &
        python3 client.py --server_address $server:43101 --cid $(($cid  + 6)) --model Net --device cpu &
        python3 client.py --server_address $server:43101 --cid $(($cid  + 7)) --model Net --device cpu &
        python3 client.py --server_address $server:43101 --cid $(($cid  + 8)) --model Net --device cpu &
        python3 client.py --server_address $server:43101 --cid $(($cid  + 9)) --model Net --device cpu &
        python3 client.py --server_address $server:43101 --cid $(($cid  + 10)) --model Net --device cpu &
        python3 client.py --server_address $server:43101 --cid $(($cid  + 11)) --model Net --device cpu &
        python3 client.py --server_address $server:43101 --cid $(($cid  + 12)) --model Net --device cpu &
        python3 client.py --server_address $server:43101 --cid $(($cid  + 13)) --model Net --device cpu &
        python3 client.py --server_address $server:43101 --cid $(($cid  + 14)) --model Net --device cpu &
        python3 client.py --server_address $server:43101 --cid $(($cid  + 15)) --model Net 
    else
        python3 client.py --server_address $server:43101 --cid $(($cid  + 0)) --model Net --device cpu &
        python3 client.py --server_address $server:43101 --cid $(($cid  + 1)) --model Net --device cpu &
        python3 client.py --server_address $server:43101 --cid $(($cid  + 2)) --model Net --device cpu &
        python3 client.py --server_address $server:43101 --cid $(($cid  + 3)) --model Net --device cpu &
        python3 client.py --server_address $server:43101 --cid $(($cid  + 4)) --model Net --device cpu &
        python3 client.py --server_address $server:43101 --cid $(($cid  + 5)) --model Net --device cpu &
        python3 client.py --server_address $server:43101 --cid $(($cid  + 6)) --model Net --device cpu &
        python3 client.py --server_address $server:43101 --cid $(($cid  + 7)) --model Net --device cpu &
        python3 client.py --server_address $server:43101 --cid $(($cid  + 8)) --model Net --device cpu &
        python3 client.py --server_address $server:43101 --cid $(($cid  + 9)) --model Net --device cpu &
        python3 client.py --server_address $server:43101 --cid $(($cid  + 10)) --model Net --device cpu &
        python3 client.py --server_address $server:43101 --cid $(($cid  + 11)) --model Net --device cpu
    fi
fi
 