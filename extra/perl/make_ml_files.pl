#! /usr/bin/perl
use Getopt::Long;
use File::Basename;

my $convert_dir;
my $training_dir;

## -- ##
sub process_one_atlas_structure {
    my ($atlas_img_file, $atlas_structure_file) = @_;
    
    print "Processing $atlas_img_file, $atlas_structure_file\n";
}

## -- ##
sub process_one_atlas {
    my ($cdir) = @_;

    my $atlas_img_file = "$cdir/img.nrrd";
    (-f $atlas_img_file) or die "Sorry, \"$atlas_img_file\" not found?";

    my $atlas_structure_dir = "$cdir/structures";
    opendir ASDIR, $atlas_structure_dir
      or die "Can't open \"$atlas_structure_dir\" for parsing";
    while (my $f = readdir(ASDIR)) {
	($f eq "." || $f eq "..") and next;
	# Temporary hack
	($f eq "BrainStem.nrrd") or next;
	$f = "$atlas_structure_dir/$f";
	(-f $f) or next;
	process_one_atlas_structure ($atlas_img_file,$f);
    }
    closedir ASDIR;
}


## Main ##
$usage = "make_ml_files.pl [options]\n";
GetOptions ("convert-dir=s" => \$convert_dir,
	    "training-dir=s" => \$training_dir)
    or die $usage;

(-d $convert_dir) or die "Sorry, convert dir $convert_dir not proper\n" . $usage;
(-d $training_dir) or die $usage;

opendir ADIR, $convert_dir or die "Can't open \"$convert_dir\" for parsing";
while (my $f = readdir(ADIR)) {
    ($f eq "." || $f eq "..") and next;
    $f = "$convert_dir/$f";
    (-d $f) or next;
    process_one_atlas ($f);
}
closedir ADIR;
