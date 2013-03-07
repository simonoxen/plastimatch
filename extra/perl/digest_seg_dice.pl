#! /usr/bin/perl

if ($#ARGV < 0) {
    die "Usage: digest_seg_dice.pl dice_file\n";
}

$seg_dice_fn = shift;

open (FP, "<$seg_dice_fn") or die "Couldn't open file $seg_dice_fn for read";
while (<FP>) {
    ($target,$reg,$structure,$_) = split (',',$_,4);
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

    $keystring = "$structure,$rho,$sigma,$minsim,$thresh";

    $dice_num{$keystring} ++;
    $dice_sum{$keystring} += $dice;


    $alt_key = "$target,$rho,$sigma,$minsim,$thresh";
    if ($structure eq "brainstem") {
	$brainstem_hash{$alt_key} = $dice;
    }
    if ($structure eq "left_eye_ball") {
	$left_eye_hash{$alt_key} = $dice;
    }
    if ($structure eq "left_parotid") {
	$left_parotid{$alt_key} = $dice;
    }
}

foreach $keystring (sort keys %brainstem_hash) {
#    print "$brainstem_hash{$keystring},$left_eye_hash{$keystring}\n";
    print "$brainstem_hash{$keystring},$left_parotid{$keystring}\n";
}


exit (0);

foreach $keystring (sort keys %dice_sum) {
    $avg_dice = $dice_sum{$keystring} / $dice_num{$keystring};
    print "$keystring: $avg_dice\n";
}
