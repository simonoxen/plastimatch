#! /usr/bin/perl

use File::Copy;
use File::Spec;
use File::Path;

#####################################################################
##     configuration variables
#####################################################################
$move_files = 0;
$want_mip = 0;
$want_min_ip = 0;
$want_ave_ip = 1;
$want_unsorted = 0;
$need_sleep = 0;

$free_breathing = 0;
$mip = 0;
$min_ip = 0;
$ave_ip = 0;
$sorted = 0;
$unsorted = 0;


#####################################################################
##     handle_directory_change
#####################################################################
sub move_or_copy {
    my ($infn,$outfn,$move_files) = @_;
    if ($move_files) {
	move ($infn, $outfn);
    } else {
	`ln -s $infn $outfn`;
    }
}

#####################################################################
##     my_mkdir
#####################################################################
sub my_mkdir {
    my ($dirname) = @_;
#    print "Trying to mkdir $dirname\n";
    mkdir $dirname;
    $need_sleep && sleep 4;
}

#####################################################################
##     find_series
#####################################################################
## Use heuristics to identify series
## GCS: These heuristics are good for scans from GE, but only if it 
## was not anonymized on GE.  Descriptions are cleared by GE anon tool.
sub find_series {
    my $series_description = shift (@_);
    if ($series_description =~ /(\d+)\%/) {
	$sno = $1;
	$sorted ++;
	return $sno;
    }
    if ($series_description =~ /contrast/i) {
	$free_breathing ++;
	return "free_breathing";
    }
    if ($series_description =~ /w\//i) {
	$free_breathing ++;
	return "free_breathing";
    }
    if ($series_description =~ /without/i) {
	$free_breathing ++;
	return "free_breathing";
    }
    if ($series_description =~ /esoph/i) {
	$free_breathing ++;
	return "free_breathing";
    }
    if ($series_description =~ /helic/i) {
	$free_breathing ++;
	return "free_breathing";
    }
    if ($series_description =~ /^MIP/) {
	$mip ++;
	return "mip"
    }
    if ($series_description =~ /^Min-IP/) {
	$min_ip ++;
	return "min-ip";
    }
    if ($series_description =~ /^Ave-IP/) {
	$ave_ip ++;
	return "ave-ip";
    }
    $unsorted ++;
    return "unsorted";
}

#####################################################################
##     MAIN
#####################################################################
if ($#ARGV == 1) {
    $indir = $ARGV[0];
    $outdir = $ARGV[1];
    $use_no_phi = 0;
} else {
    die "Usage: conquest_split.pl indir outdir\n";
}

opendir(DIR, $indir);
if ($use_no_phi) {
    @files = grep(/\-no\-phi\.v2$/,readdir(DIR));
} else {
    @files = grep(/.v2$/,readdir(DIR));
    rewinddir DIR;
    @files = (@files, grep(/.dcm$/,readdir(DIR)));
}
closedir(DIR);


foreach $file (@files) {
    $f = File::Spec->catfile($indir, $file);
    $series_description = `dcmdump -q $f | grep \"\(0008,103e\)\"`;
    chomp($series_description);

    $series_description_hash{$file} = &find_series ($series_description);
}

## Display heuristics results
print "--------------------\n";
$indir =~ /([^\\\/]*)$/;
print "DIR = $1\n";
print "--------------------\n";
print "UNSORTED:       $unsorted\n";
print "SORTED:         $sorted\n";
print "FREE_BREATHING: $free_breathing\n";
print "MIP:            $mip\n";
print "MIN IP:         $min_ip\n";
print "AVE IP:         $ave_ip\n";
print "--------------------\n";

## Make parent dir
mkpath($outdir);
$need_sleep && sleep 4;

foreach $file (@files) {
    $series = $series_description_hash{$file};
    $od = File::Spec->catfile($outdir, $series);
    my_mkdir ($od);
    $f = File::Spec->catfile($indir,$file);
    &move_or_copy ($f, $od, $move_files);
}
