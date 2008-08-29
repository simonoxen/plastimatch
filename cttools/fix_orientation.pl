$id = "0045";

$basedir = "G:\\reality\\processed-2.4.1";

$indir = "$basedir\\$id\\mha";
foreach $f (`ls $indir`) {
    chomp($f);
    $fn = "$indir\\${f}";
    print "Parsing $fn\n";
    open (FP,"+<$fn") or die "Couldn't open $fn";
    $curpos = tell(FP);
    while (<FP>) {
	if (/AnatomicalOrientation = RPI/ || /AnatomicalOrientation = \?\?\?/) {
	    seek(FP,$curpos,SEEK_SET);
	    print FP "AnatomicalOrientation = RAI";
	    break;
	}
	$curpos = tell(FP);
    }
    close FP;
}
