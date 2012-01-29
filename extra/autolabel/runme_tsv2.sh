train=1
test=0

if test $train != 0; then
    plastimatch autolabel-train \
	--task tsv2 \
	--output-dir tsv \
	--input /home/gsharp/Dropbox/autolabel/gold/ 
#	--input mini
fi

if test $test != 0; then
    plastimatch autolabel \
	--task tsv2 \
	--network-dir tsv \
	--output-fcsv tsv/tsv2_out.fcsv \
	--input /home/gsharp/Dropbox/autolabel/rider-pilot/0022_10109.nrrd
#	--input /home/gsharp/Dropbox/autolabel/rider-pilot/0021_9771.nrrd \
#	--input /home/gsharp/Dropbox/autolabel/gold/rider-pilot/0001_144.nrrd \
fi
