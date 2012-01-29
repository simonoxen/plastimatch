train=0
test=1

if test $train != 0; then
    plastimatch autolabel-train \
	--task tsv1 \
	--output-dir tsv \
	--input /home/gsharp/Dropbox/autolabel/gold/ 
#	--input mini 
fi

if test $test != 0; then
    plastimatch autolabel \
	--eac \
	--task tsv1 \
	--network-dir tsv \
	--output-csv tsv/tsv1_out.csv \
	--input /home/gsharp/Dropbox/autolabel/rider-pilot/0022_10109.nrrd
#	--input /home/gsharp/Dropbox/autolabel/rider-pilot/0021_9771.nrrd 
#	--input /home/gsharp/Dropbox/autolabel/gold/rider-pilot/0001_144.nrrd \
fi
