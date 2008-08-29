$infile = shift;
$outfile = shift;
$outfile || die "Usage: dcm_contour_points.pl infile_contour outfile\n";
(-f $infile) || die "Error: input file not found\n";

sub grab_value () {
    ($_) = @_;
    chomp;
    s/^.*\[//;
    s/\].*//;
    return $_;
}

## First pass: find contour set ordering within file
$cmd = "dcmdump +L $infile";
open D1, "$cmd|";
open DO, ">$outfile";

#gets the Series Instance UID
while(<D1>){
	if(/\(3006,0010/){
		last;
	}
}
while(<D1>){
	if(/\(0020,000e/){
		$CT_Series_UID= &grab_value($_);
		last;
	}
}

$roi_number = -1;
while (<D1>) {
    #if (/\(0020,000d/){
	#$studyID= &grab_value($_);
    #}elsif(/\(3006,0022/) {

    if(/\(3006,0022/) {
	$roi_number = &grab_value($_);
    } elsif (/\(3006,0026/) {
	$roi_name = &grab_value($_);
	$roi_name_hash{$roi_number} = $roi_name;
	$roi_number = -1;
    } elsif (/\(3006,0039/) {
	last;
    }
}
$order = 1;
$roi_color = "255\\255\\127";
while (<D1>) {
    if (/\(3006,0080/) {
	last;
    } elsif (/\(3006,002a/) {
	$roi_color = &grab_value($_);
    } elsif (/\(3006,0084/) {
	$roi_number = &grab_value($_);
	$roi_order{$order} = "$roi_color|$roi_name_hash{$roi_number}";
	$roi_color = "255\\255\\127";
	$order = $order + 1;
    }
}
close D1;

## Second pass: write contours
open D1, "$cmd|";
open DO, ">$outfile";
print DO "SERIES_CT_UID ";
print DO "$CT_Series_UID\n";
for $key (sort keys %roi_order) {
    $val = $roi_order{$key};
    ($roi_color,$roi_name) = split (/\|/,$val);
    print DO "$key $roi_color $roi_name\n";
}
print DO "END_OF_ROI_NAMES\n";
while (<D1>) {
    if (/\(3006,0040/) {
	last;
    }
}

#########  GCS Aug 20, 2008
#########  GE no longer includes contour number and attached contour.
$roi_no = -1;
while (<D1>) {
    if (/\(3006,0080/) {
	last;
    } elsif (/\(3006,0040/) {   # Start next structure
	$order = $order + 1;
    } elsif(/\(0008,1155/){
	$UID= &grab_value($_);
    } elsif (/\(3006,0044/) {
	$ct = &grab_value($_);
    } elsif (/\(3006,0046/) {
	$np = &grab_value($_);
    } elsif (/\(3006,0050/) {
	$cp = &grab_value($_);
	push @contours, "$ct $np $UID $cp";
    } elsif (/\(3006,0084/) {
	$roi_no = &grab_value($_);
	while ($_ = pop @contours) {
	    print DO "$roi_no $_\n";
	}
    }
}
close D1;
close DO;
