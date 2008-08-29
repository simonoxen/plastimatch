## Usage: merge_contour_points file prefix file prefix
##   Writes to stdout

if ($#ARGV != 3) {
    die "Usage: merge_contour_points file prefix file prefix\n";
}

$file1 = shift;
$prefix1 = shift;
$file2 = shift;
$prefix2 = shift;
(-f $file1) || die "Can't find file: $file1\n";
(-f $file2) || die "Can't find file: $file2\n";

open F1, "<$file1";
open F2, "<$file2";
while (<F1>) {
    if (/END_OF_ROI_NAMES/) {
	last;
    }
    /^([0-9]+)\s+(\S+)\s+(.*)/;
    $roi_no = $1;
    $roi_color = $2;
    $roi_name = $3;

    $roi_name = $prefix1 . $roi_name;
    print "$roi_no $roi_color $roi_name\n";
}
$f1_num_rois = $roi_no;
while (<F2>) {
    if (/END_OF_ROI_NAMES/) {
	print;
	last;
    }
    chomp;
    /^([0-9]+)\s+(\S+)\s+(.*)/;
    $roi_no = $1;
    $roi_color = $2;
    $roi_name = $3;

    $roi_no = $roi_no + $f1_num_rois;
    $roi_name = $prefix2 . $roi_name;
    print "$roi_no $roi_color $roi_name\n";
}
while (<F1>) {
    print;
}
while (<F2>) {
    chomp($roi_no = $_);
    $roi_no =~ s/ .*//;
    $roi_no = $roi_no + $f1_num_rois;
    $_ =~ s/^[0-9]*//;
    print $roi_no . $_;
}
close F1;
close F2;
