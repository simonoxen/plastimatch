#!/bin/bash
rm -rf out*
# Make sure there's a plastimatch binary available in the same directory as this file. If plastimatch is installed replace ./plastimatch with plastimatch 
./plastimatch synth --dim "512" --pattern rect --output rect.mha 
./plastimatch synth --dim "512" --pattern sphere --output sphere.mha
for threads in {1,2,4,8,16,32}
	do
	echo "Number of threads = $threads"
		for i in {1..20} #Number of iterations is 20, if changed please also change in the python script line number 236
			do
			echo "Iteration number $i/20"
			echo "Iteration number $i" >>output_$threads.csv

			for j in {25,30,35,40,45,50,60,65,75,90}
				do
					#echo $j
					sed -i "14s/XYZ/$j/g" parms.txt	
					export OMP_NUM_THREADS=$threads
					#echo $OMP_NUM_THREADS
					./plastimatch register parms.txt > out1
					sed -r 's/.* ([0-9]+\.*[0-9]*).*?/\1/' out1 > out2
					sed -n '37'p out2 >> output_$threads.csv
					sed -i "14s/$j/XYZ/g" parms.txt
		
				done
		done
	done
./test.py
#xdg-open le_openmp2.eps
#xdg-open le_openmp_21.eps
rm -rf rect.mha
rm -rf sphere.mha

