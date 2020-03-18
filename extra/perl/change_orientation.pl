#!/usr/bin/perl

use File::Copy qw(copy move);

$push_to_mim = 0;
$overwrite_for = 1;

#$dicom_dir = "/PHShome/gcs6/shared/ben-1/019-01-14";
#$dicom_dir = "/PHShome/gcs6/shared/ben-1/LPcom_tt000_v2";
#$dicom_dir = "/PHShome/gcs6/shared/ben-1/LPcom_tt000_v2-stripped";
#$dicom_dir = "/PHShome/gcs6/shared/ben-1/test";
#$dicom_dir = "/PHShome/gcs6/shared/ben-1/LPcommissSRS_BB";
#$dicom_dir = "/PHShome/gcs6/shared/ben-1/LPcommis_mornqa";
#$dicom_dir = "/PHShome/gcs6/shared/ben-1/LPcom_01";
#$dicom_dir = "/PHShome/gcs6/shared/ben-1/LPcom_02";
#$dicom_dir = "/PHShome/gcs6/shared/ben-1/PLOGOS_D45";
#$dicom_dir = "/PHShome/gcs6/shared/ben-1/LPcommis_AbdomenLarge";
#$dicom_dir = "/PHShome/gcs6/shared/ben-1/LPcom_mornqa";
#$dicom_dir = "/PHShome/gcs6/shared/ben-1/LPcom_logos_v1";
#$dicom_dir = "/PHShome/gcs6/shared/ben-1/017-08-01";
#$dicom_dir = "/PHShome/gcs6/shared/ben-1/T4D-QA";
#$dicom_dir = "/PHShome/gcs6/shared/ben-1/T1";
#$dicom_dir = "/PHShome/gcs6/shared/ben-1/var_thick";
#$dicom_dir = "/PHShome/gcs6/shared/ben-1/2D^CAL^PHANTOM";
$dicom_dir = "/PHShome/gcs6/shared/ben-1/tmp";

$new_name = "";
$new_id = "";
$new_birth_date = "";
$new_sex = "";
$new_name = "LPcom_01^PBS";
$new_id = "LPcom_01";
$new_birth_date = "20180101";
$new_sex = "O";
#$new_name = "LPcom_02^PBS";
#$new_id = "LPcom_02";
#$new_birth_date = "20180725";
#$new_sex = "O";

$new_series_description = "Medcom phantom";
#$new_series_description = "Var Thick";

$new_patient_position = "";
$new_image_orientation = "";

#$new_patient_position = "FFS";
#$new_image_orientation = "-1\\0\\0\\0\\-1\\0";

#######################################################################################

$dcmdir = "/PHShome/gcs6/shared/ben-1/out";
$txtdir = "/PHShome/gcs6/shared/ben-1/txt";
(not -d $dcmdir) and mkdir "$dcmdir";
(not -d $txtdir) and mkdir "$txtdir";

unlink glob "$dcmdir/*";
unlink glob "$txtdir/*";

$overwrite_for_uid = `dicom_uid`;
chomp($overwrite_for_uid);

%uid_map = {};

sub get_value {
    my ($value) = @_;
    if ($value =~ /.*\[(.*)\]/) {
	$value = $1;
    } else {
	die "Parse error.";
    }
    return $value
}

sub reorient {
    my ($x, $y, $z) = @_;

#    return ($x + 0.06, $y - 1.5, $z + 355.4);
#    return ($x - 0.91, $y + 61.17 - 6.8, $z - 9.88);
#    return ($x + 0.79, $y + 70.97, $z + 499.39);
#    return (-$x + 10, -$y + 20, $z + 30);
#    return ($x, $y, $z-500);
#    return ($x + 0.800000, $y - 3.900000, $z + 499.399994);
#    return ($x + 0.34, $y - 1.24, $z + 500.61);
    return ($x - 0.38, $y - 3.88, $z + 90);
#    return ($x, $y, $z);
}

sub change_isocenter {
    my ($x, $y, $z) = @_;

#    return (10, -14, 485);
#    return (0, 0, 1000);
#    return (10, 20, 30 - 10);
#    return (-$x, -$y, $z);
#    return ($x, $y, $z);
    return (0, 0, 0);
}

sub process_file {

    my ($dcm_file) = @_;

    $cmd = "dcmdump -q +W raw +L $dcm_file > dump.txt";
    print ("$cmd\n");
    system ($cmd);

    $dump_in = "dump.txt";
    $dump_out = "dump_out.txt";

    open FIN, "<$dump_in";
    open FOUT, ">$dump_out";
    $sop_instance_uid = "";
    while (<FIN>) {
	if (/^\s*\(([^)]*)\)/) {
	    $key = $1;
	} else {
	    next;
	}
	if ($key eq "0002,0003" or $key eq "0008,0018" or $key eq "0020,000d"
	    or $key eq "0020,000e" or $key eq "0020,0052" or $key eq "0008,1155"
	    or $key eq "3006,0024") {
	    $value = get_value ($_);
	    if (($key eq "0020,0052" or $key eq "3006,0024") and $overwrite_for) {
		print FOUT "($key) UI [$overwrite_for_uid]\n";
		next;
	    }
	    if (not exists ($uid_map {$value})) {
		$new_uid = `dicom_uid`;
		chomp ($new_uid);
		$uid_map{$value} = $new_uid;
	    }
	    if ($key eq "0008,0018") {
		$sop_instance_uid = $uid_map{$value};
	    }
	    print FOUT "($key) UI [$uid_map{$value}]\n";
	    next;
	}
	if ($key eq "0008,103e" and $new_series_description ne "") {
	    print FOUT "($key) LO [$new_series_description]\n";
	    next;
	}
	if ($key eq "0010,0010" and $new_name ne "") {
	    print FOUT "($key) PN [$new_name] # PatientName\n";
	    next;
	}
	if ($key eq "0010,0020" and $new_id ne "") {
	    print FOUT "($key) LO [$new_id] # PatientID\n";
	    next;
	}
	if ($key eq "0010,0030" and $new_birth_date ne "") {
	    print FOUT "($key) DA [$new_birth_date] # PatientBirthDate\n";
	    next;
	}
	if ($key eq "0010,0040" and $new_sex ne "") {
	    print FOUT "($key) CS [$new_sex] # PatientSex\n";
	    next;
	}
	if ($key eq "0018,5100" and $new_patient_position ne "") {
	    print FOUT "($key) CS [$new_patient_position] # PatientPosition\n";
	    next;
	}
	if ($key eq "0020,0037" and $new_image_orientation ne "") {
	    print FOUT "($key) DS [$new_image_orientation] # ImageOrientationPatient\n";
	    next;
	}
	if ($key eq "300a,0002" and $new_series_description ne "") {
	    print FOUT "($key) SH [$new_series_description]\n";
	    next;
	}
	if ($key eq "0020,0032") {
	    $value = get_value ($_);
	    my ($x,$y,$z,$rest) = split (/\\/,$value,3);
	    ($x,$y,$z) = reorient ($x,$y,$z);
	    print FOUT "($key) DS [$x\\$y\\$z]\n";
	    next;
	}
	if ($key eq "300a,0082" or $key eq "300a,012c") {
	    $value = get_value ($_);
	    my ($x,$y,$z,$rest) = split (/\\/,$value,3);
	    ($x,$y,$z) = change_isocenter ($x,$y,$z);
	    print FOUT "($key) DS [$x\\$y\\$z]\n";
	    next;
	}
	# ContourPoints
	if ($key eq "3006,0050") {
	    $value = get_value ($_);
	    $outvalue = "";
	    while (true) {
		my ($x,$y,$z,$rest) = split (/\\/,$value,4);
		($x,$y,$z) = reorient ($x,$y,$z);
		if ($outvalue ne "") {
		    $outvalue = "$outvalue\\";
		}
		$outvalue = "${outvalue}$x\\$y\\$z";
		if ($rest eq "") {
		    last;
		}
		$value = $rest;
	    }
	    print FOUT "($key) DS [$outvalue]\n";
	}
	print FOUT $_;
    }
    close FIN;
    close FOUT;

    if (${sop_instance_uid} eq "") {
	die "Error, no SOP Instance UID";
    }

    $new_dump_out = "${txtdir}/${sop_instance_uid}.txt";
    move ("${dump_out}", "$new_dump_out");
    $cmd = "dump2dcm +l 500000 ${new_dump_out} ${dcmdir}/${sop_instance_uid}.dcm";
    system ($cmd);
}

@dcm_files = ();
opendir (DIR, $dicom_dir);
while ($fn = readdir (DIR)) {
    next if $fn =~ /^\./;
    push @dcm_files, "$dicom_dir/$fn";
}
close (DIR);

for my $dcm_file (@dcm_files) {
    process_file ($dcm_file);
}

if ($push_to_mim) {
    $cmd = "push_mim.sh ${dcmdir}";
    system ($cmd);
}
