#! /usr/bin/perl

if ($#ARGV < 0) {
    die "Usage: digest_seg_dice.pl dice_file\n";
}

$seg_dice_fn = shift;

$max_hd = 200;

open (FP, "<$seg_dice_fn") or die "Couldn't open file $seg_dice_fn for read";
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
    $key_0 = "$reg";
    $key_0_dice_num{$key_0} ++;
    $key_0_dice_sum{$key_0} += $dice;
    $key_0_hd95_sum{$key_0} += $hd95;

    # For segmentation evaluation
    if ($thresh > -1) {
	$key_1 = "$structure,$rho,$sigma,$minsim,$thresh";
	$key_1_dice_num{$key_1} ++;
	$key_1_dice_sum{$key_1} += $dice;
	$key_1_hd95_sum{$key_1} += $hd95;
    }

    $key_2 = "$target,$rho,$sigma,$minsim,$thresh";
    if ($structure eq "brainstem") {
	$brainstem_hash{$key_2} = $dice;
    }
    if ($structure eq "left_eye_ball") {
	$left_eye_hash{$key_2} = $dice;
    }
    if ($structure eq "left_parotid") {
	$left_parotid{$key_2} = $dice;
    }
}

# foreach $keystring (sort keys %brainstem_hash) {
# #    print "$brainstem_hash{$keystring},$left_eye_hash{$keystring}\n";
#     print "$brainstem_hash{$keystring},$left_parotid{$keystring}\n";
# }


# exit (0);


foreach $k (sort keys %key_0_dice_sum) {
    $avg_dice = $key_0_dice_sum{$k} / $key_0_dice_num{$k};
    $avg_hd95 = $key_0_hd95_sum{$k} / $key_0_dice_num{$k};
    print "$k,$avg_dice,$avg_hd95\n";
}

foreach $k (sort keys %key_1_dice_sum) {
    $avg_dice = $key_1_dice_sum{$k} / $key_1_dice_num{$k};
    $avg_hd95 = $key_1_hd95_sum{$k} / $key_1_dice_num{$k};
    print "$k,$avg_dice,$avg_hd95\n";
}
