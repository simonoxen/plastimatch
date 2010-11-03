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
##     MAIN
#####################################################################
if ($#ARGV == 0) {
    $indir = $ARGV[0];
} else {
    die "Usage: query_103.pl indir\n";
}

opendir(DIR, $indir);
@indir_files = readdir(DIR);
closedir(DIR);

foreach $indir_entry (@indir_files) {
    ($indir_entry eq ".." || $indir_entry eq ".") && next;
    $subdir = File::Spec->catfile ($indir, $indir_entry);
    (-d $subdir) || next;

    opendir(DIR, $subdir);
    @files = readdir(DIR);
    closedir(DIR);

    #print "dir: $subdir\n";
    foreach $dicom_file (@files) {
	(-d $dicom_file) && next;
	$dicom_file = File::Spec->catfile($subdir, $dicom_file);

	# Skip if it is a scout
	$_ = `dcmdump $dicom_file | grep \"0008,0008\" | tail -n 1`;
	$_ =~ /LOCALIZER/ && last;

	# Find if this CT matches the structure set
	$_ = `dcmdump $dicom_file | grep SeriesInstanceUID | tail -n 1`;
	$_ =~ /^.*\[(.*)\].*/;
	$_ = $1;
	if ($indir_entry eq "0103") {
	    $ss_file = $dicom_file;
	    $referenced_uid = $_;
	} else {
	    $uid_hash{$indir_entry} = $_;
	}
	last;
    }
}

mkdir "mha";
for $key (keys %uid_hash) {

    # Check if it matches to create README
    if ($uid_hash{$key} eq $referenced_uid) {
	$referenced_dir = File::Spec->catfile ($indir, $key);
	open README, ">mha/README";
	print README "Structure set matches $key\n";
	close README;
    }

    # Convert mha file
    $in_subdir = File::Spec->catfile ($indir, $key);
    $cmd = "plastimatch convert --input $in_subdir --output mha/$key.mha";
    print "$cmd\n";
    print `$cmd`;
}

# Convert structure set volumes
$cmd = "plastimatch convert --input $ss_file --dicom-dir $referenced_dir --output-prefix mha/ss/ss";
print "$cmd\n";
print `$cmd`;
