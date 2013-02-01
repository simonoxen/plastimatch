#! /usr/bin/perl

if ($#ARGV < 0) {
    die "Usage: digest_seg_dice.pl dice_file\n";
}

$seg_dice_fn = shift;

open (FP, "<$seg_dice_fn") or die "Couldn't open file $seg_dice_fn for read";
while (<FP>) {
    ($a1,$reg,$structure,$rho,$sigma,$thresh,$dice,$junk) = split (',',$_);
    if ($dice eq "") { next; }

    $keystring = "$structure,$rho,$sigma,$thresh";

    $dice_num{$keystring} ++;
    $dice_sum{$keystring} += $dice;
}

foreach $keystring (sort keys %dice_sum) {
    $avg_dice = $dice_sum{$keystring} / $dice_num{$keystring};
    print "$keystring: $avg_dice\n";
}
