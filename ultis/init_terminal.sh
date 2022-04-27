#!/bin/bash
read -p 'numero de terminales juanp :' num_terminal

for i in $(seq 1 $num_terminal);do
    echo $i
    if (($i == $num_terminal))
    then 
        sshpass -p p.123 ssh -Y pocampo@172.19.18.35
    else  
        gnome-terminal --tab --command="sshpass -p p.123 ssh -Y pocampo@172.19.18.35"
    fi
done

