die "Usage: make_dicomrt.pl uids.txt contours.cxt\n" 
  unless $#ARGV >= 1;

use POSIX;
use File::Basename;

## -----------------------------------------------------------------
##  Global settings
## -----------------------------------------------------------------

## Dicomrt header strings
use lib dirname($0);
do "make_dicomrt_inc.pl" || die "Can't load include file: make_dicom_inc.pl";

## -----------------------------------------------------------------
##  Structure file parsing routines
## -----------------------------------------------------------------
sub parse_cxt_format {
    my ($fn) = @_;

    ## This is the structure we will fill in
    $structure_set = { };
    $structure_set->{header} = { };
    $structure_set->{structures} = [ ];
    $ss_structures = $structure_set->{structures};

    ## This code was copied and edited from contour_points_to_cxt.pl.  
    ## It needs to be re-engineered into a common subroutine for both scripts.
    $series_ct_uid = "Unknown";  # For future work
    $same_study_set = 1;
    $have_roi_names = 0;

    ## Read header
    open CF, "<$fn" || die "CXT file not found: $fn\n";
    while (<CF>) {
	chomp;
	if (/SERIES_CT_UID/) {
	    ($junk, $series_ct_contour) = split;
	    if ($series_ct_contour ne $series_ct_uid) {
		print "SERIES_CT_UID_CT: $series_ct_uid\n";
		print "SERIES_CT_UID_CN: $series_ct_contour\n";
		warn "Warning: contours and ct are from different study sets\n";
		$same_study_set = 0;
	    }
	    $structure_set->{header}->{ct_series_uid} = $series_ct_contour;
	} else {
	    if (/ROI_NAMES/) {
		$have_roi_names = 1;
	    } elsif ($have_roi_names) {
		if (!/END_OF_ROI_NAMES/) {
		    ($structure_no,$color,$name) = split /\|/;
		    $ss_structures->[$structure_no]->{color} = $color;
		    $ss_structures->[$structure_no]->{name} = $name;
		    $ss_structures->[$structure_no]->{contours} = [ ];

		    ## $structure_color_hash{$structure} = $color;
		    ## GE must replace spaces with underscores (?)
		    ## $name =~ s/ /_/g;
		    ## $structure_names_hash{$structure} = $name;
		}
	    }
	}
	last if /END_OF_ROI_NAMES/;
    }

#    @roi_sort = sort { $a <=> $b } keys %structure_names_hash;
#    while ($i = shift @roi_sort) {
#	push @roi_names, $structure_names_hash{$i};
#	push @roi_colors, $structure_colors_hash{$i};
#    }
#    push @$structure_names, @roi_names;
#    push @$structure_colors, @roi_colors;

    $old_struct = -1;
    while (<CF>) {
	($structure_no, 
	 $contour_thickness, 
	 $num_points, 
	 $slice_index, 
	 $uid_contour, 
	 $points) = split /\|/;

        push @{ $ss_structures->[$structure_no]->{contours} }, $_;
    }
    close CF;
    return $structure_set;
}


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
#parse_cxt_format ($cxt_fn, \@structures, \@structure_names, 
#		  \@structure_colors);

$structure_set = parse_cxt_format ($cxt_fn);
$ss_structures = $structure_set->{structures};

for $i (0..$#{$ss_structures}) {
    $s = $ss_structures->[$i];
    next if not $s->{name};
    print ">> $i $s->{name} $#{$s->{contours}}\n";
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

##########################################################################
## All of the following must be correct for GE AdvW to associate the 
## structures with the CT.
##########################################################################
## $patient_name = "Anonymous";
## $patient_id = "0028";
## $series_instance_uid = "1.2.840.113619.2.55.1.1762853477.1996.1155908038.536.103";
## $series_instance_uid = "2.16.840.1.114362.1.90609.1196125526343.847.103";
## $study_instance_uid = "1.2.840.113619.2.55.1.1762853477.1996.1155908038.536";
## $study_id = "9400";
$patient_name = "NPC_panel";
$patient_id = "ANON42627";
$study_id = "ANON26726";
$patient_sex = "M";

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
  $patient_name,
  $patient_id,
  $patient_sex,
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

#    $structure = $structures[$i];
#    $color = "255\\0\\0";
#    $color = $structure_colors[$i];
    $color = $s->{color};
    printf OUT $subhead_103_part4, $color, $i;
    $j = 1;
    for $contour (@{$s->{contours}}) {
	# Convert points to a string
#	$pts = "";
#	for $pt (@{$contour}) {
#	    ($x, $y, $z) = split ' ', $pt;
# 	    if ($z < 0 || $z > $#slices) {
#		printf "Warning: skipping contour with index %d ($i)\n", $z;
#		last;
#	    }
#	    $x = ($x * $pixel_x) + $off_x;
#	    $y = ($y * $pixel_y) + $off_y;
#	    $z_loc = $slices[$z];
#	    if ($pts) {
#		$pts = $pts . "\\$x\\$y\\$z_loc";
#	    } else {
#		$pts = "$x\\$y\\$z_loc";
#	    }
#	}
#	printf OUT $item_103_part4_without_ac, 
#	  $slice_UIDs{$z_loc}, $#{$contour}+1, $j, $pts;

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

