#! /usr/bin/perl

sub digest_file {
    my ($fn, $reghash, $seghash) = @_;
    open (FP, "<$fn") or die "Couldn't open file $fn for read";
    while (<FP>) {
	($target,$_) = split (',',$_,2);
	$reg = "";
	$structure = "";
	$rho = -1;
	$sigma = -1;
	$minsim = -1;
	$thresh = -1;
	$dice = -1;
	$hd = -1;
	$bhd = -1;
	$hd95 = -1;
	$bhd95 = -1;
	$ahd = -1;
	$abhd = -1;
	while (($val,$_) = split (',',$_,2)) {
	    if ($val =~ /reg=(.*)/) {
		$reg = $1;
		next;
	    }
	    if ($val =~ /struct=(.*)/) {
		$structure = $1;
		next;
	    }
	    if ($val =~ /rho=(.*)/) {
		$rho = $1;
		next;
	    }
	    if ($val =~ /sigma=(.*)/) {
		$sigma = $1;
		next;
	    }
	    if (($val =~ /minsum=(.*)/) || ($val =~ /minsim=(.*)/)) {
		$minsim = $1;
		next;
	    }
	    if ($val =~ /thresh=(.*)/) {
		$thresh = $1;
		next;
	    }
	    if ($val =~ /dice=(.*)/) {
		$dice = $1;
		next;
	    }
	    if ($val =~ /^hd=(.*)/) {
		$hd = $1;
		next;
	    }
	    if ($val =~ /^bhd=(.*)/) {
		$bhd = $1;
		next;
	    }
	    if ($val =~ /^95hd=(.*)/) {
		$hd95 = $1;
		if ($hd95 > $max_hd) {
		    $hd95 = $max_hd;
		}
		next;
	    }
	    if ($val =~ /^95bhd=(.*)/) {
		$bhd95 = $1;
		next;
	    }
	    if ($val =~ /^ahd=(.*)/) {
		$ahd = $1;
		next;
	    }
	    if ($val =~ /^abhd=(.*)/) {
		$abhd = $1;
		next;
	    }
	}

	# For registration evaluation
	$reg_key = "$reg";
	if (exists $reghash->{$reg_key}) {
	    ($num,$dice_sum,$hd95_sum) = split (',', $reghash->{$reg_key});
	    $num++;
	    $dice_sum += $dice;
	    $hd95_sum += $hd95;
	} else {
	    $num = 1;
	    $dice_sum = $dice;
	    $hd95_sum = $hd95;
	}
	$reghash->{$reg_key} = "$num,$dice_sum,$hd95_sum";
	
	# For segmentation evaluation
	if ($thresh > -1) {
	    $seg_key = "$structure,$rho,$sigma,$minsim,$thresh";
	    if (exists $seghash->{$seg_key}) {
		($num,$dice_sum,$hd95_sum) = split (',', $seghash->{$seg_key});
		$num++;
		$dice_sum += $dice;
		$hd95_sum += $hd95;
	    } else {
		$num = 1;
		$dice_sum = $dice;
		$hd95_sum = $hd95;
	    }
	    $seghash->{$seg_key} = "$num,$dice_sum,$hd95_sum";
	}
	
	# $key_2 = "$target,$rho,$sigma,$minsim,$thresh";
	# if ($structure eq "brainstem") {
	#     $brainstem_hash{$key_2} = $dice;
	# }
	# if ($structure eq "left_eye_ball") {
	#     $left_eye_hash{$key_2} = $dice;
	# }
	# if ($structure eq "left_parotid") {
	#     $left_parotid{$key_2} = $dice;
	# }
    }
}


###########################################################################
##  MAIN
###########################################################################

if ($#ARGV < 0) {
    die "Usage: digest_mabs_stats.pl dice_file | training_dir\n";
}

$dice_source = shift;

$max_hd = 200;

%reghash = ();
%seghash = ();

if (-f $dice_source) {
    digest_file ($dice_source, \%reghash, \%seghash);
} elsif (-d $dice_source) {
    opendir DIR, $dice_source or die "Can't open \"$dice_source\" for parsing";
    while (my $f = readdir(DIR)) {
	($f eq "." || $f eq "..") and next;
	$fn = "$dice_source/$f/reg_dice.csv";
	if (-f "$fn") {
	    digest_file ($fn, \%reghash, \%seghash);
	} 
	$fn = "$dice_source/$f/seg_dice.csv";
	if (-f "$fn") {
	    digest_file ($fn, \%reghash, \%seghash);
	} 
    }
    closedir DIR;
} else {
    die "Can't open \"$dice_source\" for parsing";
}

$best_reg = "";
$best_reg_score = 0;
foreach $reg (sort keys %reghash) {
    ($num,$dice_sum,$hd95_sum) = split (',', $reghash{$reg});
    $avg_dice = $dice_sum / $num;
    $avg_hd95 = $hd95_sum / $num;
    print "$reg,$avg_dice,$avg_hd95\n";
    if ($avg_dice > $best_reg_score) {
	$best_reg_score = $avg_dice;
	$best_reg = $reg;
    }
}
if (-d $dice_source && $best_reg ne "") {
    ## Update training file
    $fn = "$dice_source/optimization_result_reg.txt";
    open FP, ">$fn";
    print FP "[OPTIMIZATION_RESULT]\nregistration=$best_reg\n";
    close FP;
}

$best_seg = "";
$best_seg_score = 0;
foreach $seg (sort keys %seghash) {
    ($num,$dice_sum,$hd95_sum) = split (',', $seghash{$seg});
    $avg_dice = $dice_sum / $num;
    $avg_hd95 = $hd95_sum / $num;
    print "$seg,$avg_dice,$avg_hd95\n";
}
if (-d $dice_source && $best_seg ne "") {
    ## Update training file
    $fn = "$dice_source/optimization_result_seg.txt";
    open FP, ">$fn";
    print FP "[OPTIMIZATION_RESULT]\nregistration=$best_seg\n";
    close FP;
}

# exit (0);
# foreach $keystring (sort keys %brainstem_hash) {
# #    print "$brainstem_hash{$keystring},$left_eye_hash{$keystring}\n";
#     print "$brainstem_hash{$keystring},$left_parotid{$keystring}\n";
# }
