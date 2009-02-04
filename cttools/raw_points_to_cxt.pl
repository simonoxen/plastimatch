die "Usage: raw_points_to_cxt.pl uids.txt output.cxt point_file [point_file ...]\n" 
  unless $#ARGV >= 1;

use POSIX;
use File::Basename;

## -----------------------------------------------------------------
##  Global settings
## -----------------------------------------------------------------

## Dicomrt header strings
use lib dirname($0);

$plastimatch_uid_prefix = "1.2.826.0.1.3680043.8.274.1";
$dump_in_fn = "dump_in.txt";
$dcm_out_file = "dump.dcm";
$dump_out_fn = "dump_out.txt";

## -----------------------------------------------------------------
##  Structure file parsing routines
## -----------------------------------------------------------------
sub parse_marta_format {
    my ($fn, $structures, $structure_names) = @_;
    open CF, "<$fn";
    while (<CF>) {
	chomp;
	if (/^NaN/) {
	    if ($pts) {
		push @{$contours}, $pts;
		undef $pts;
	    }
	    next;
	}
	push @{$pts}, $_;
    }
    close CF;

    if ($pts) {
	push @{$contours}, $pts;
	undef $pts;
    }
    push @$structures, $contours;
    undef $contours;
    
    $structure_name = $fn;
    if ($fn =~ m/[^_]*_([^\.]*)_physcoord.txt$/) {
	$structure_name = $1;
    } else {
	$structure_name = "Structure $i";
	$i ++;
    }
    push @$structure_names, $structure_name;
}

## -----------------------------------------------------------------
##  Load UID file
## -----------------------------------------------------------------
$uid_fn = shift;
(-f $uid_fn) || die "Error: uid file \"$uid_fn\" not found\n";
open UIF, "<$uid_fn" || die "Can't open uid file \"$uid_fn\" for read\n";
while (<UIF>)
{
    next if /^#/;
    ($junk, $off_x, $off_y, $off_z, $nr_X, $nr_Y, $pixel_y, $pixel_x, $slice_uid, $study_img_uid, $series_img_uid, $for_img_uid) = split;
    $slice_UIDs{$off_z}=$slice_uid;
}
close UIF;
@slices = sort { $a <=> $b } keys %slice_UIDs;


## -----------------------------------------------------------------
##  Output filename
## -----------------------------------------------------------------
$output_fn = shift;


## -----------------------------------------------------------------
##  Load contour files
## -----------------------------------------------------------------
@structures = ( );
@structure_names = ( );
$i = 1;
while ($fn = shift) {
    parse_marta_format ($fn, \@structures, \@structure_names);
}


## -----------------------------------------------------------------
##  Write CXT
## -----------------------------------------------------------------
open OF, ">$output_fn";
print OF "SERIES_CT_UID Unknown\n";
print OF "ROI_NAMES\n";
$i = 1;
for $structure_name (@structure_names) {
    print OF "$i|255\\0\\0|$structure_name\n";
    $i++;
}
print OF "END_OF_ROI_NAMES\n";

$i = 1;
for $structure (@structures) {
    for $contour (@{$structure}) {
	print OF "$i||$#{$contour}|||";
	$first = 1;
 	for $pt (@{$contour}) {
	    chomp($pt);
	    $pt =~ s/^\s*//;
	    $pt =~ s/\s*$//;
	    $pt =~ s/\s+/\\/g;
	    if ($first) {
		print OF "$pt";
	    } else {
		print OF "\\$pt";
	    }
	    $first = 0;
  	}
	print OF "\n";
    }
    $i++;
}
close OF;


exit;

($sec,$min,$hour,$mday,$mon,$year,$wday,$yday,$isdst) 
  = localtime(time);
$date = sprintf("%04d%02d%02d",$year + 1900, $mon + 1, $mday);
$time = sprintf("%02d%02d%02d", $hour, $min, $sec);

($sysname, $nodename, $release, $version, $machine) = POSIX::uname();
#print "$sysname | $nodename | $release | $version | $machine\n";

$instance_creation_date = $date;
$instance_creation_time = $time;
$instance_creator_uid = $plastimatch_uid_prefix;
$sop_instance_uid = $plastimatch_uid_prefix . ".1";
$station_name = $nodename;
$patient_name = "Anonymous";
$patient_id = "0028";
$software_version = "<EXPERIMENTAL>";
$study_instance_uid = "1.2.840.113619.2.55.1.1762853477.1996.1155908038.536";
## $series_instance_uid = $plastimatch_uid_prefix . ".2";
$series_instance_uid = "1.2.840.113619.2.55.1.1762853477.1996.1155908038.536.103";
$study_id = "9400";
$series_number = "103";
$instance_number = "1";
$structure_set_label = "Test";
$structure_set_name = "Test";
$structure_set_date = $date;
$structure_set_time = $time;

## Write dicomrt part 1
open OUT, ">$dump_in_fn";
printf OUT $head_103_part1,
  $instance_creation_date,
  $instance_creation_time,
  $instance_creator_uid,
  $sop_instance_uid,
  $station_name,
  $patient_name,
  $patient_id,
##  $software_version,
  $study_instance_uid,
  $series_instance_uid,
  $study_id,
  $series_number,
  $instance_number
  ;

## Write dicomrt part 2 from image uid's
printf OUT $head_103_part2, 
  $structure_set_label,
  $structure_set_name,
  $structure_set_date,
  $structure_set_time,
  $for_img_uid,
  $study_img_uid,
  $series_img_uid;
for $i (0..$#slices) {
    printf OUT $item_103_part2, $slice_UIDs{$slices[$i]};
}
print OUT $foot_103_part2;

## Write dicomrt part 3 from contour file
print OUT $head_103_part3;
for $i (0..$#structures) {
  printf OUT $item_103_part3, 
    $i + 1, 
    $for_img_uid, 
    $structure_name[$i];
  ;
}
print OUT $foot_103_part3;

## Rebuild part 4 from contour file
print OUT $head_103_part4;
for $i (0..$#structures) {
    $structure = $structures[$i];
    $color = "255\\0\\0";
    printf OUT $subhead_103_part4, $color, $i + 1;
    $j = 1;
    for $contour (@{$structure}) {
	# Convert points to a string
	$pts = "";
	for $pt (@{$contour}) {
	    ($x, $y, $z) = split ' ', $pt;
 	    if ($z < 0 || $z > $#slices) {
		printf "Warning: skipping contour with index %d ($i)\n", $z;
		last;
	    }
	    $x = ($x * $pixel_x) + $off_x;
	    $y = ($y * $pixel_y) + $off_y;
	    $z_loc = $slices[$z];
	    if ($pts) {
		$pts = $pts . "\\$x\\$y\\$z_loc";
	    } else {
		$pts = "$x\\$y\\$z_loc";
	    }
	}
	printf OUT $item_103_part4_without_ac, 
	  $slice_UIDs{$z_loc}, $#{$contour}+1, $j, $pts;
        $j = $j + 1;
    }
    printf OUT $subfoot_103_part4, $i + 1;
}
print OUT $foot_103_part4;

## Rebuild part 5 from contour file
print OUT $head_103_part5;
for $i (0..$#structures) {
    printf OUT $item_103_part5, 
      $i + 1,
      $i + 1,
      $structure_name[$i];
}
print OUT $foot_103_part5;
close OUT;

print `dump2dcm +E -e -g +l 524288 $dump_in_fn $dcm_out_file`;
#print `dcmdump +L +R 512 $dcm_out_file > $dump_out_fn`;

