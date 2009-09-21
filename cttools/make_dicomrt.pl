#!/usr/bin/perl
#######################################################
# This function creates a DicomRT structure set from 
# a UIDS file and a CXT file.  The UIDS file is used 
# only for creating the complete list of slices.  
# All other UIDS come from the CXT file.
#######################################################
die "Usage: make_dicomrt.pl uids.txt contours.cxt\n" 
  unless $#ARGV >= 1;

## -----------------------------------------------------------------
##  Global settings
## -----------------------------------------------------------------

## Load include files
use POSIX;
use File::Basename;
use lib dirname($0);
do "parse_cxt.pl" || die "Can't load include file: parse_cxt.pl";
do "make_dicomrt_inc.pl" || die "Can't load include file: make_dicom_inc.pl";

## Hard coded to .1 sub-range for software development
$plastimatch_uid_prefix = "1.2.826.0.1.3680043.8.274.1.1";
$uids_fn = shift;
$cxt_fn = shift;
$dump_in_fn = "dump_in.txt";
$dcm_out_file = "dump.dcm";
$dump_out_fn = "dump_out.txt";

## -----------------------------------------------------------------
##  Load UID file
## -----------------------------------------------------------------
(-f $uids_fn) || die "Error: uid file \"$uids_fn\" not found\n";
open UIF, "<$uids_fn" || die "Can't open uid file \"$uids_fn\" for read\n";
while (<UIF>)
{
    next if /^#/;
    ($junk, $off_x, $off_y, $off_z, $nr_X, $nr_Y, $pixel_y, $pixel_x, $slice_uid, $study_img_uid, $series_img_uid, $for_img_uid) = split;
    $slice_UIDs{$off_z}=$slice_uid;
}
close UIF;
@slices = sort { $a <=> $b } keys %slice_UIDs;

## -----------------------------------------------------------------
##  Load contour files
## -----------------------------------------------------------------
@structures = ( );
@structure_names = ( );
@structure_colors = ( );

## Load CXT file
$structure_set = parse_cxt_format ($cxt_fn);
$ss_header = $structure_set->{header};
$ss_structures = $structure_set->{structures};

for $i (0..$#{$ss_structures}) {
    $s = $ss_structures->[$i];
    next if not $s->{name};
    print ">> $i $s->{name} $#{$s->{contours}}\n";
}

## -----------------------------------------------------------------
##  Convert undesirable characters in structure names for GE
## -----------------------------------------------------------------
for $i (0..$#{$ss_structures}) {
  $s = $ss_structures->[$i];
  next if not $s->{name};

  $s->{name} =~ s/ /_/g;
  $s->{name} =~ s/\+/x/g;
}


($sec,$min,$hour,$mday,$mon,$year,$wday,$yday,$isdst) 
  = localtime(time);
$date = sprintf("%04d%02d%02d",$year + 1900, $mon + 1, $mday);
$time = sprintf("%02d%02d%02d", $hour, $min, $sec);

($sysname, $nodename, $release, $version, $machine) = POSIX::uname();
#print "$sysname | $nodename | $release | $version | $machine\n";

$instance_creation_date = $date;
$instance_creation_time = $time;
$station_name = $nodename;
$software_version = "<EXPERIMENTAL>";

#####################################
## These have to be filled in manually
#####################################
# $patient_name = "NPC_panel";
# $patient_id = "ANON42627";
# $study_id = "ANON26726";
# $patient_sex = "M";

# $patient_name = "OP_panel";
# $patient_id = "ANON65526";
# $study_id = "ANON26726";
# $patient_sex = "M";

# $patient_name = "Glot_panel";
# $patient_id = "ANON74245";
# $study_id = "ANON26726";
# $patient_sex = "M";

# $patient_name = "BUCKLEY^JOHN";
# $patient_id = "4352161";
# $study_id = "8280";
# $patient_sex = "M";



## Create Dicom unique identifiers
$instance_creator_uid = $plastimatch_uid_prefix;
$sop_instance_uid = `dicom_uid $plastimatch_uid_prefix`;
chomp($sop_instance_uid);
$series_instance_uid = `dicom_uid $plastimatch_uid_prefix`;
chomp($series_instance_uid);

$study_instance_uid = $study_img_uid;
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
  $ss_header->{patient_name},
  $ss_header->{patient_id},
  $ss_header->{patient_sex},
##  $software_version,
  $study_instance_uid,
  $series_instance_uid,
  $ss_header->{study_id},
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
for $i (0..$#{$ss_structures}) {
  $s = $ss_structures->[$i];
  next if not $s->{name};

  printf OUT $item_103_part3, 
    $i, 
    $for_img_uid, 
    $s->{name};
  ;
}
print OUT $foot_103_part3;

## Rebuild part 4 from contour file
print OUT $head_103_part4;
for $i (0..$#{$ss_structures}) {
    $s = $ss_structures->[$i];
    next if not $s->{name};

    $color = $s->{color};
    printf OUT $subhead_103_part4, $color, $i;
    $j = 1;
    for $contour (@{$s->{contours}}) {
	($structure_no, 
	 $contour_thickness, 
	 $num_points, 
	 $slice_index, 
	 $uid_contour, 
	 $points) = split /\|/, $contour;
	chomp ($points);

	printf OUT $item_103_part4_without_ac, 
	  $uid_contour, $num_points, $j, $points;

        $j = $j + 1;
    }
    printf OUT $subfoot_103_part4, $i;
}
print OUT $foot_103_part4;

## Rebuild part 5 from contour file
print OUT $head_103_part5;
for $i (0..$#{$ss_structures}) {
    $s = $ss_structures->[$i];
    next if not $s->{name};

    printf OUT $item_103_part5, 
      $i, 
      $i, 
      $s->{name};
}
print OUT $foot_103_part5;
close OUT;

print `dump2dcm +E -e -g +l 524288 $dump_in_fn $dcm_out_file`;
#print `dcmdump +L +R 512 $dcm_out_file > $dump_out_fn`;

