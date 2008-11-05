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
$cmd = "dcmdump +L +R 512 $infile";
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
	$ct_series_uid = &grab_value($_);
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
	$ordered_rois{$order} = "$roi_color|$roi_name_hash{$roi_number}";
	$roi_order{$roi_number} = $order;
	$roi_color = "255\\255\\127";
	$order = $order + 1;
    }
}
close D1;

## Second pass: write contours
open D1, "$cmd|";
open DO, ">$outfile";
print DO "SERIES_CT_UID $ct_series_uid\n";
print DO "ROI_NAMES\n";
for $key (sort {$a <=> $b} keys %ordered_rois) {
    $val = $ordered_rois{$key};
    ($roi_color,$roi_name) = split (/\|/,$val);
    print DO "$key|$roi_color|$roi_name\n";
}
print DO "END_OF_ROI_NAMES\n";
while (<D1>) {
    if (/\(3006,0040/) {
	last;
    }
}

## Create CXT file
$roi_no = -1;
while (<D1>) {
    if (/\(3006,0080/) {
	last;
    } elsif (/\(3006,0040/) {   # Start of ContourSequence
	$order = $order + 1;
    } elsif(/\(0008,1155/){     # UID of associated CT slice
	$UID= &grab_value($_);
    } elsif (/\(3006,0044/) {   # Contour slab thickness
	$ct = &grab_value($_);
    } elsif (/\(3006,0046/) {   # Number of contour points
	$np = &grab_value($_);
    } elsif (/\(3006,0050/) {   # Contour data
	$cp = &grab_value($_);
	push @contours, "$ct|$np||$UID|$cp";
    } elsif (/\(3006,0084/) {   # Referenced ROI number
	$roi_no = &grab_value($_);
	$roi_order_no = $roi_order{$roi_no};
	while ($_ = pop @contours) {
	    print DO "$roi_order_no|$_\n";
	}
    }
}
close D1;
close DO;
