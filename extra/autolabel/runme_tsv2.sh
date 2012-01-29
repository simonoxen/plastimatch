train=0
test=1

if test $train != 0; then
    plastimatch autolabel-train \
	--task tsv2 \
	--input /home/gsharp/Dropbox/autolabel/gold/ \
	--output-csv tsv2/tsv2.csv \
	--output-net tsv2/tsv2.net \
	--output-tsacc tsv2/tsv2.txt
#	--input mini \
fi

if test $test != 0; then
    plastimatch autolabel \
	--task tsv2 \
	--input /home/gsharp/Dropbox/autolabel/rider-pilot/0021_9771.nrrd \
	--network tsv2/tsv2.net \
	--output tsv2/tsv2_a.csv
#	--input /home/gsharp/Dropbox/autolabel/gold/rider-pilot/0001_144.nrrd \
fi
