#######################################################
# This function merges information from a UIDS file 
# into a CXT file.  Specifically, slice locations, pixel 
# spacing, and offset are added to the CXT header, and 
# CXT contours are labeled with the matching slice number.
#######################################################
$cxt_in_fn = shift;
$uid_fn = shift;
$outfile= shift;
$outfile || die "Usage: contour_points_to_cxt.pl contours.txt uids.txt outfile\n";
(-f $cxt_in_fn) || die "Error: input file not found ($cxt_in_fn)\n";
(-f $uid_fn) || die "Error: input file not found ($uid_fn)\n";

## Load include files
use File::Basename;
use lib dirname($0);
do "parse_cxt.pl" || die "Can't load include file: parse_cxt.pl";

## Load UID file
open FN2, "<$uid_fn";
while (<FN2>)
{
    next if /^#/;
    ($junk, $off_X, $off_Y, $off_Z, $nr_X, $nr_Y, $pixel_X, $pixel_Y, $slice_UID, $junk, $series_CT_UID, $junk) = split;
    $slice_UIDs{$off_Z}=$slice_UID;
}
close FN2;

## Check Z spacing
@slices = sort { $a <=> $b } keys %slice_UIDs;
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

## Build reverse lookup table
for $k (0..$#slices) {
    $slice_no{$slice_UIDs{$slices[$k]}}=$k;
    #print "$slice_no{$slice_UIDs{$slices[$k]}}\n";
}

## Find slice with minimum Z 
$min_z = $slices[0];
for $i (1..$#slices) {
    if ($slices[$i] < $min_z) {
	$min_z = $slices[$i];
    }
}
$no_slices=$#slices+1;

## Load input CXT file
$structure_set = parse_cxt_format ($cxt_in_fn);
$ss_header = $structure_set->{header};
$ss_structures = $structure_set->{structures};


## Write CXT header
open GO, ">$outfile";
print GO "SERIES_CT_UID $series_CT_UID\n";
print GO "OFFSET $off_X $off_Y $min_z\n";
print GO "DIMENSION $nr_X $nr_Y $no_slices\n";
print GO "SPACING $pixel_X $pixel_Y $difference[0]\n";

## GCS FIX: These should come from the UIDS file, but this is 
## not yet implemented
print GO "PATIENT_NAME $ss_header->{patient_name}\n";
print GO "PATIENT_ID $ss_header->{patient_id}\n";
print GO "STUDY_ID $ss_header->{study_id}\n";
print GO "PATIENT_SEX $ss_header->{patient_sex}\n";

## Write list of structures
print GO "ROI_NAMES\n";
for $i (0..$#{$ss_structures}) {
    $s = $ss_structures->[$i];
    next if not $s->{name};
    print GO "$i|$s->{color}|$s->{name}\n";
}
#print GO "END_OF_ROI_NAMES\n";


## Write list of polylines
for $i (0..$#{$ss_structures}) {
    $s = $ss_structures->[$i];
    next if not $s->{name};

    for $contour (@{$s->{contours}}) {
	($structure_no, 
	 $contour_thickness, 
	 $num_points, 
	 $slice_index_to_be_replaced, 
	 $uid_contour, 
	 $points) = split /\|/, $contour;

	## If matching with same study set, find slice number 
	## by matching the UID
	if ($same_study_set) {
	    if (!exists $slice_no{$uid_contour}) {
		print "Error. No matching image UID found.";
		print "UID=$uid_contour\n";
		die;
	    }
	    $sno = $slice_no{$uid_contour};
	}

	## If matching with different study set, find slice number 
	## by matching the Z location, and change the UID
	else {
	    $contour_z_loc = $points;
	    $contour_z_loc =~ s/^[^\\]*\\[^\\]*\\//;
	    $contour_z_loc =~ s/\\.*//;
	    $best_dist = abs($slices[0]-$contour_z_loc);
	    $best_slice = 0;
	    $uid_contour = $slice_UIDs{$slices[0]};
	    for $i (0..$#slices) {
		$dist=abs($slices[$i]-$contour_z_loc);
		if ($dist < $best_dist) {
		    $best_dist = $dist;
		    $best_slice = $i;
		    $uid_contour = $slice_UIDs{$slices[$i]};
		}
	    }
	    $sno = $best_slice;
	}
	print GO "$structure_no|$contour_thickness|$num_points|$sno|$uid_contour|$points";
    }
}

close GO;
