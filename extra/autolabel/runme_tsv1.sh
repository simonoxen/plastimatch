train=0
test=1

if test $train != 0; then
    plastimatch autolabel-train \
	--task tsv1 \
	--input /home/gsharp/Dropbox/autolabel/gold/ \
	--output-csv tsv1/tsv1.csv \
	--output-net tsv1/tsv1.net \
	--output-tsacc tsv1/tsv1.txt
#    --input mini \
fi

if test $test != 0; then
    plastimatch autolabel \
	--eac \
	--task tsv1 \
	--input /home/gsharp/Dropbox/autolabel/rider-pilot/0021_9771.nrrd \
	--network tsv1/tsv1.net \
	--output tsv1/tsv1_a.csv
#	--input /home/gsharp/Dropbox/autolabel/gold/rider-pilot/0001_144.nrrd \
fi

