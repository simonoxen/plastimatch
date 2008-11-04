## -----------------------------------------------------------------
##  Structure file parsing routines
## -----------------------------------------------------------------
sub parse_cxt_format {
    my ($fn) = @_;

    ## This is the structure we will fill in
    $structure_set = { };
    $structure_set->{header} = { };
    $structure_set->{structures} = [ ];
    $structure_set->{contours} = [ ];

    ## This code was copied and edited from contour_points_to_cxt.pl.  
    ## It needs to be re-engineered into a common subroutine for both scripts.
    $series_CT_UID = "Unknown";  # For future work
    $same_study_set = 1;
    $have_roi_names = 0;

    ## Read header
    open CF, "<$fn" || die "CXT file not found: $fn\n";
    while (<CF>) {
	chomp;
	if (/SERIES_CT_UID/) {
	    ($junk, $series_CT_contour) = split;
	    if ($series_CT_contour ne $series_CT_UID) {
		print "SERIES_CT_UID_CT: $series_CT_UID\n";
		print "SERIES_CT_UID_CN: $series_CT_contour\n";
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
		    $structure_set->{structures}->[$structure_no]->{color} = $color;
		    $structure_set->{structures}->[$structure_no]->{name} = $name;
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

	if ($old_struct != $structure_no) {
	    if ($contours) {
		#push @$structures, $contours;
		push $structure_set->{contours}->[$old_struct], $contours;
		undef $contours;
	    }
	}
	$old_struct = $structure_no;
	push @{$contours}, $_;
    }
    close CF;
    if ($contours) {
	#push @$structures, $contours;
	push $structure_set->{contours}->[$old_struct], $contours;
	undef $contours;
    }
}

1;   ## Successfully included file

1;
