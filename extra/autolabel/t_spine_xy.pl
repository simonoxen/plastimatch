# Special thanks to perlmonks
# http://www.perlmonks.org/?node_id=136482
# http://www.perlmonks.org/?node_id=584507

# Not quite finished.  Need to do the following:
#  -> Convert slicer coords to LPS
#  -> Make libsvm format
#  -> Figure out single or double output learning

use File::Copy;
use File::Find;
use File::Path;
use File::Spec;
use File::Spec::Functions;

$dropbox_dir = "$ENV{HOME}/Dropbox";
$in_base = "$dropbox_dir/autolabel/gold";
$out_base = "$ENV{HOME}/scratch/autolabel/t_spine_xy";

# Do it!
process_recursive ($in_base);

sub parse_fcsv {
    ($fn) = @_;
    @locs = ();

    open FP, "<$fn";
    while (<FP>) {
	next if /^#/;
	chomp;
	($label,$x,$y,$z,$rest) = split (/,/, $_, 5);
	push @locs, "$label,$x,$y,$z";
    }
    close FP;

    return @locs;
}

sub process_file {
    my $fn = shift;
    return if (not $fn =~ /\.fcsv$/);

    print "Trying to process $fn\n";

    my $base = $fn;
    $base =~ s/\.fcsv$//;
    my $nrrd = "${base}.nrrd";
    return if (! -f $nrrd);

    @locs = parse_fcsv ($fn);
    for $loc (@locs) {
	($label,$x,$y,$z) = split (/,/, $loc, 4);

	$out_mhd = $out_base . "_${label}.mhd";
	$out_raw = $out_base . "_${label}.raw";
	$cmd = "plastimatch thumbnail "
	  . "--input \"$nrrd\" --output \"$out_mhd\" "
	    . "--thumbnail-dim 16 --thumbnail-spacing 25.0 --slice-loc $z";
	print "$cmd\n";
	print `$cmd`;

	open FH, "<$out_raw";
	binmode FH;
	print "$x $y ";
	while (read (FH, $buf, 4)) {
	    print unpack ('f', $buf) . " ";
	}
	close FH;
	print "\n";
	exit;
    }
}

sub process_recursive {
    my $path = shift;

    print "Recursive on $path\n";

    # Get list of files, not including ".", ".."
    opendir (DIR, $path) or
      die "Can't open directory $path: $!\n";
    my @files = grep { !/^\.{1,2}$/ } readdir (DIR);
    closedir (DIR);

    # Change @files from relative to absolute path
    @files = map { $path . '/' . $_ } @files;

    for (@files) {
	if (-d $_) {
	    # Recurse on directory
            process_recursive ($_);
	} else {
	    # Process file
	    process_file ($_);
	}
    }
}
