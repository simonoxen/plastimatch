#! /usr/bin/perl

$indir = "/home/gcs6/shared/rtog0522-resampled";
$outdir = "/home/gcs6/shared/rtog0522-gabor";

opendir (my $dh, "$indir") || die;

while (readdir $dh) {
    m/^0522/ || next;
    for my $i (0 .. 11) {
	$cmd = sprintf "plastimatch filter --output \"$outdir/$_/gabor_%02d.nrrd\" --pattern gabor --gabor-k-fib \"$i 12\" \"$indir/$_/img.nrrd\"", $i;
	print "$cmd\n";
	print `$cmd`;
    }
}

closedir $dh;
