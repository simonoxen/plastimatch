#######################################################
# This function merges information from the CT and 
# the contours.  Specifically, slice locations, pixel 
# spacing, and offset are added to the header, and 
# contours are labeled with the matching slice number.
#######################################################
$contour_fn = shift;
$uid_fn = shift;
$outfile= shift;
$outfile || die "Usage: contour_points_to_cxt.pl contours.txt uids.txt outfile\n";
(-f $contour_fn) || die "Error: input file not found ($contour_fn)\n";
(-f $uid_fn) || die "Error: input file not found ($uid_fn)\n";

open FN2, "<$uid_fn";

while (<FN2>)
{
    next if /^#/;
    ($junk, $off_X, $off_Y, $off_Z, $nr_X, $nr_Y, $pixel_X, $pixel_Y, $slice_UID, $junk, $series_CT_UID, $junk) = split;
    $slice_UIDs{$off_Z}=$slice_UID;
}

#for $key (keys %slice_UIDs) {
#    print "$key $slice_UIDs{$key}\n";
#}

@slices = sort { $a <=> $b } keys %slice_UIDs;

#for $j(0..$#slices) {
#    print "$slices[$j] \n";
#}

for $i (0..$#slices-1) {
    $difference[$i]=$slices[$i+1]-$slices[$i];
    if ($i !=0){
	$d = abs ($difference[$i-1] - $difference[0]);
	if ($d > 0.001) {
	    printf "Uneven slice separation: [%d] %g [%d] %g\n", 
	      $i-1, $difference[$i-1], 0, $difference[0];
	}
    }
}

for $k (0..$#slices) {
    $slice_no{$slice_UIDs{$slices[$k]}}=$k;
    #print "$slice_no{$slice_UIDs{$slices[$k]}}\n";
}

open FN1, "<$contour_fn";
open GO, ">$outfile";

$same_study_set = 1;
while (<FN1>)
{
    #print "Hello!";
    chomp;
    if (/SERIES_CT_UID/) {
	($junk, $series_CT_contour)=split;	
	if ($series_CT_contour ne $series_CT_UID) {
	    print "SERIES_CT_UID_CT: $series_CT_UID\n";
	    print "SERIES_CT_UID_CN: $series_CT_contour\n";
	    warn "Warning: contours and ct are from different study sets\n";
	    $same_study_set = 0;
	}
    } else {
	if (!/END_OF_ROI_NAMES/) {
	    ($structure,$color,$name) = split ' ', $_, 3;
	    $structure_names{$structure} = $name;
	    $structure_colors{$structure} = $color;
	}
    }
    last if /END_OF_ROI_NAMES/;
}
#close FN1;

## Find slice with minimum Z 
$min_z = $slices[0];
for $i (1..$#slices) {
    if ($slices[$i] < $min_z) {
	$min_z = $slices[$i];
    }
}

$no_slices=$#slices+1;
@roi_names = sort { $a <=> $b } keys %structure_names;

print GO "SERIES_CT_UID $series_CT_contour\n";
print GO "OFFSET $off_X $off_Y $min_z\n";
print GO "DIMENSION $nr_X $nr_Y $no_slices\n";
print GO "SPACING $pixel_X $pixel_Y $difference[0]\n";
print GO "ROI_NAMES\n";

for $r (0..$#roi_names) {
    $num=$r+1;
    print GO "$num $structure_colors{$roi_names[$r]} $structure_names{$roi_names[$r]}\n";
}

#for $key (keys %structure_names) {
#   print GO "$key $structure_names{$key}\n";
#}
print GO "END_OF_ROI_NAMES\n";

#print "SERIES_CT_UID: $series_CT_contour\n";
#print "OFFSET: $off_X $off_Y $slices[$#slices]\n";
#print "DIMENSION: $nr_X $nr_Y $no_slices\n";
#print "SPACING: $pixel_X $pixel_Y $difference[0]\n";

#open FN1, "<$contour_fn";

while (<FN1>) {
    ($structure_no, 
     $contour_thickness, 
     $num_points, 
     $slice_index_to_be_replaced, 
     $uid_contour, 
     $points) = split /\|/;

    ## If matching with same study set, index by UID
    if ($same_study_set) {
	if (!exists $slice_no{$uid_contour}) {
	    print "Error. No matching image UID found.";
	    print "UID=$uid_contour\n";
	    die;
	}
	$sno = $slice_no{$uid_contour};
    }

    ## If matching with different study set, index by Z location
    else {
	$contour_z_loc = $points;
	$contour_z_loc =~ s/^[^\\]*\\[^\\]*\\//;
	$contour_z_loc =~ s/\\.*//;
	$best_dist = abs($slices[0]-$contour_z_loc);
	$best_slice = 0;
	for $i (0..$#slices) {
	    $dist=abs($slices[$i]-$contour_z_loc);
	    if ($dist < $best_dist) {
		$best_dist = $dist;
		$best_slice = $i;
	    }
	}
	$sno = $best_slice;
    }
    print GO "$structure_no|$contour_thickness|$num_points|$sno|$uid_contour|$points";
}

close FN1;
close FN2;
close GO;
