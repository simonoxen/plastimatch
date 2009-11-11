#! /usr/bin/perl

###### The number at the end of the output directory 
######    is the "StudyID" from the dicom header
###### Find it like this:
######    dcmdump `ls | head -n 1` | grep StudyID
######
######    105	Adv Sim RT Plans
######    103	Adv Sim RT Structure Sets
######    107	Advantage Fusion RT Structure Sets

use File::Copy;
use File::Spec;
use File::Path;

#####################################################################
##     configuration variables
#####################################################################
$move_files = 1;
$want_mip = 1;
$want_min_ip = 1;
$want_ave_ip = 1;
$want_unsorted = 1;

##$free_breathing = 3;
$free_breathing = -1;
$mip = -1;
$min_ip = -1;
$ave_ip = -1;
$unsorted = -1;
$s103 = -1;
$s105 = -1;
$s107 = -1;


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
}
## @files = grep(/\-no\-phi\.v2$/,readdir(DIR));
closedir(DIR);

$f = File::Spec->catfile($indir, $files[1]);
$studyid = `dcmdump $f | grep StudyID`;
chomp($studyid);
$studyid =~ s/[^[]*\[//;
$studyid =~ s/].*//;

foreach $file (@files) {
    $file =~ /_([^_]*)/;
    $series = $1;
    if (!defined $series_hash{$series}) {
	$series_hash{$series} = 1;
	$f = File::Spec->catfile($indir, $file);
	$series_description = `dcmdump $f | grep \"\(0008,103e\)\"`;
	chomp($series_description);
	$series_description =~ s/[^[]*\[//;
	$series_description =~ s/].*//;
	
	print "$series $series_description\n";
	$series_description_hash{$series} = $series_description;
    } else {
	$series_hash{$series} = $series_hash{$series} + 1;
    }
}

## Use heuristics to identify series
foreach $k (keys %series_description_hash) {
    if ($series_hash{$k} > 500) {
	$unsorted = $k;
    } 
    elsif ($series_description_hash{$k} =~ /^T=(\d*)\%/) {
	$sno = $1 / 10;
	$sorted[$sno] = $k;
    }
    elsif ($series_description_hash{$k} =~ /contrast/i) {
	$free_breathing = $k;
    }
    elsif ($series_description_hash{$k} =~ /w\//i) {
	$free_breathing = $k;
    }
    elsif ($series_description_hash{$k} =~ /without/i) {
	$free_breathing = $k;
    }
    elsif ($series_description_hash{$k} =~ /esoph/i) {
	$free_breathing = $k;
    }
    elsif ($series_description_hash{$k} =~ /helic/i) {
	$free_breathing = $k;
    }
    elsif ($series_description_hash{$k} =~ /^MIP/) {
	$mip = $k;
    }
    elsif ($series_description_hash{$k} =~ /^Min-IP/) {
	$min_ip = $k;
    }
    elsif ($series_description_hash{$k} =~ /^Ave-IP/) {
	$ave_ip = $k;
    }
    elsif ($k == 103) {
	$s103 = $k;
    }
    elsif ($k == 105) {
	$s105 = $k;
    }
    elsif ($k == 107) {
	$s107 = $k;
    }
}

## Display heuristics results
print "--------------------\n";
$indir =~ /([^\\\/]*)$/;
print "DIR = $1\n";
print "StudyID = $studyid\n";
print "--------------------\n";
print "UNSORTED:       $unsorted\n";
print "SORTED:         ";
foreach (0..9) { print $sorted[$_] . " "; }
print "\n";
print "FREE_BREATHING: $free_breathing\n";
print "MIP:            $mip\n";
print "MIN IP:         $min_ip\n";
print "AVE IP:         $ave_ip\n";
print "STRUCTS:        $s103 $s105 $s107\n";
print "--------------------\n";

## Move files
if (!-d $outdir) {
    mkpath($outdir);
}

if ($free_breathing > 0) {
    print "Moving free breathing\n";
    $od = File::Spec->catfile($outdir, "free-breathing");
    mkdir $od;
    @move_files = grep(/_${free_breathing}_/,@files);
    for $file (@move_files) {
	$f = File::Spec->catfile($indir,$file);
	&move_or_copy ($f, $od, $move_files);
    }
}

if ($s103 > 0) {
    print "Moving 103\n";
    $od = File::Spec->catfile($outdir, "103");
    mkdir $od;
    @move_files = grep(/_${s103}_/,@files);
    for $file (@move_files) {
	$f = File::Spec->catfile($indir,$file);
	&move_or_copy ($f, $od, $move_files);
    }
}

if ($s105 > 0) {
    print "Moving 105\n";
    $od = File::Spec->catfile($outdir, "105");
    mkdir $od;
    @move_files = grep(/_${s105}_/,@files);
    for $file (@move_files) {
	$f = File::Spec->catfile($indir,$file);
	&move_or_copy ($f, $od, $move_files);
    }
}

if ($s107 > 0) {
    print "Moving 107\n";
    $od = File::Spec->catfile($outdir, "107");
    mkdir $od;
    @move_files = grep(/_${s107}_/,@files);
    for $file (@move_files) {
	$f = File::Spec->catfile($indir,$file);
	&move_or_copy ($f, $od, $move_files);
    }
}

for $i (0..9) {
    print "Moving t$i\n";
    $od = File::Spec->catfile($outdir, "t$i");
    mkdir $od;
    $s = $sorted[$i];
    @move_files = grep(/_${s}_/,@files);
    for $file (@move_files) {
	$f = File::Spec->catfile($indir,$file);
	&move_or_copy ($f, $od, $move_files);
    }
}

if ($want_unsorted && $unsorted > 0) {
    print "Moving unsorted\n";
    $od = File::Spec->catfile($outdir, "unsorted");
    mkdir $od;
    @move_files = grep(/_${unsorted}_/,@files);
    for $file (@move_files) {
	$f = File::Spec->catfile($indir,$file);
	&move_or_copy ($f, $od, $move_files);
    }
}

if ($want_mip && $mip > 0) {
    print "Moving mip\n";
    $od = File::Spec->catfile($outdir, "mip");
    mkdir $od;
    @move_files = grep(/_${mip}_/,@files);
    for $file (@move_files) {
	$f = File::Spec->catfile($indir,$file);
	&move_or_copy ($f, $od, $move_files);
    }
}

if ($want_min_ip && $min_ip > 0) {
    print "Moving min ip\n";
    $od = File::Spec->catfile($outdir, "min_ip");
    mkdir $od;
    @move_files = grep(/_${min_ip}_/,@files);
    for $file (@move_files) {
	$f = File::Spec->catfile($indir,$file);
	&move_or_copy ($f, $od, $move_files);
    }
}

if ($want_ave_ip && $ave_ip > 0) {
    print "Moving ave ip\n";
    $od = File::Spec->catfile($outdir, "ave_ip");
    mkdir $od;
    @move_files = grep(/_${ave_ip}_/,@files);
    for $file (@move_files) {
	$f = File::Spec->catfile($indir,$file);
	&move_or_copy ($f, $od, $move_files);
    }
}

