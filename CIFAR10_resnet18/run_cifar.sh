#!/bin/bash

echo $1
cid=$(($1-1))
cid=$(($cid * 9))
echo $(($cid  + 1))
if [[ $1 -eq 0 ]]
then 
    echo $1
    hostname -I > ip_cifar_res.txt
    server="127.0.0.1"
    while IFS=' ' read -r ip rest
    do 
        echo "$ip"
        server="$ip"
    done <ip_cifar_res.txt
    python3 server.py --server_address $server:43122 --rounds 150 --min_num_clients 96 --min_sample_size 1 --model ResNet18

else 
    sleep 20
    echo $1
    server="127.0.0.1"
    while IFS=' ' read -r ip rest
    do 
        echo "$ip"
        server="$ip"
    done <ip_cifar_res.txt

    if [[ $cid -eq 90 ]]
    then
        python3 client.py --server_address $server:43122 --cid $(($cid  + 0)) --model ResNet18 --device cpu &
        python3 client.py --server_address $server:43122 --cid $(($cid  + 1)) --model ResNet18 --device cpu &
        python3 client.py --server_address $server:43122 --cid $(($cid  + 2)) --model ResNet18 --device cpu &
        python3 client.py --server_address $server:43122 --cid $(($cid  + 3)) --model ResNet18 --device cpu &
        python3 client.py --server_address $server:43122 --cid $(($cid  + 4)) --model ResNet18 --device cpu &
        python3 client.py --server_address $server:43122 --cid $(($cid  + 5)) --model ResNet18 --device cpu
    else
        python3 client.py --server_address $server:43122 --cid $(($cid  + 0)) --model ResNet18 --device cpu &
        python3 client.py --server_address $server:43122 --cid $(($cid  + 1)) --model ResNet18 --device cpu &
        python3 client.py --server_address $server:43122 --cid $(($cid  + 2)) --model ResNet18 --device cpu &
        python3 client.py --server_address $server:43122 --cid $(($cid  + 3)) --model ResNet18 --device cpu &
        python3 client.py --server_address $server:43122 --cid $(($cid  + 4)) --model ResNet18 --device cpu &
        python3 client.py --server_address $server:43122 --cid $(($cid  + 5)) --model ResNet18 --device cpu &
        python3 client.py --server_address $server:43122 --cid $(($cid  + 6)) --model ResNet18 --device cpu &
        python3 client.py --server_address $server:43122 --cid $(($cid  + 7)) --model ResNet18 --device cpu &
        python3 client.py --server_address $server:43122 --cid $(($cid  + 8)) --model ResNet18 --device cpu 
    fi
fi
 