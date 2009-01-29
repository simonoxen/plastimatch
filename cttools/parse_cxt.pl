## -----------------------------------------------------------------
##  Structure file parsing routines
## -----------------------------------------------------------------
sub parse_cxt_format {
    my ($fn) = @_;

    ## This is the data structure
    $structure_set = { };
    $structure_set->{header} = { };
    $structure_set->{structures} = [ ];
    $ss_structures = $structure_set->{structures};

    ## The series_ct_uid should be passed as an input argument for checking
    $series_ct_uid = "Unknown";  # For future work
    $same_study_set = 1;
    $have_roi_names = 0;

    ## Read header
    open CF, "<$fn" || die "CXT file not found: $fn\n";
    while (<CF>) {
	chomp;
	if (/^SERIES_CT_UID/) {
	    ($junk, $series_ct_contour) = split;
	    if ($series_ct_contour ne $series_ct_uid) {
		print "SERIES_CT_UID_CT: $series_ct_uid\n";
		print "SERIES_CT_UID_CN: $series_ct_contour\n";
		warn "Warning: contours and ct are from different study sets\n";
		$same_study_set = 0;
	    }
	    $structure_set->{header}->{ct_series_uid} = $series_ct_contour;
	} 
	elsif (/^PATIENT_NAME/) {
	    ($junk, $structure_set->{header}->{patient_name}) = split;
	}
	elsif (/^PATIENT_ID/) {
	    ($junk, $structure_set->{header}->{patient_id}) = split;
	}
	elsif (/^STUDY_ID/) {
	    ($junk, $structure_set->{header}->{study_id}) = split;
	}
	elsif (/^PATIENT_SEX/) {
	    ($junk, $structure_set->{header}->{patient_sex}) = split;
	}
	else {
	    if (/^ROI_NAMES/) {
		$have_roi_names = 1;
	    } elsif ($have_roi_names) {
		if (!/END_OF_ROI_NAMES/) {
		    ($structure_no,$color,$name) = split /\|/;
		    $ss_structures->[$structure_no]->{color} = $color;
		    $ss_structures->[$structure_no]->{name} = $name;
		    $ss_structures->[$structure_no]->{contours} = [ ];
		}
	    }
	}
	last if /END_OF_ROI_NAMES/;
    }

    ## Read contour points
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

    ## Return results
    return $structure_set;
}

1;   ## Successfully included file
